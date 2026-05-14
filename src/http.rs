//! Embedded HTTP server: serves `graph.html` + the cached profile / insight
//! / index JSON, plus proxies `/api/*` to local Ollama. Pure `std::net`,
//! one thread per connection. Bound to 127.0.0.1 — local-only by design.

// ─── built-in HTTP server (static files + Ollama proxy) ───
//
// Serves graph.html plus the cached profile / insight / index data, and proxies
// /api/* to the local Ollama. Same-origin means no CORS dance, no `python -m http.server`,
// no `OLLAMA_ORIGINS=*`. Pure std::net, multi-threaded, ~250 lines.

pub(crate) const HTTP_PORT_DEFAULT: u16 = 8765;

pub(crate) fn http_serve(port: u16, root_dir: String) {
    use std::net::TcpListener;
    let addr = format!("127.0.0.1:{}", port);
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("\nfailed to bind {}: {}", addr, e);
            eprintln!("(another process already listening? set AUSPEX_PORT=NNNN)");
            return;
        }
    };
    eprintln!("\n──────────────────────────────────────");
    eprintln!("  serving at  http://localhost:{}/", port);
    eprintln!("  Ctrl-C to stop");
    eprintln!("──────────────────────────────────────");
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let root = root_dir.clone();
                std::thread::spawn(move || {
                    let _ = handle_request(s, &root);
                });
            }
            Err(e) => eprintln!("accept error: {}", e),
        }
    }
}

pub(crate) fn handle_request(mut stream: std::net::TcpStream, root: &str) -> std::io::Result<()> {
    use std::io::Read;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(std::time::Duration::from_secs(600)))?;

    // read until end of headers (CRLF CRLF)
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut tmp = [0u8; 4096];
    let header_end;
    loop {
        let n = match stream.read(&mut tmp) {
            Ok(0) => return Ok(()),
            Ok(n) => n,
            Err(_) => return Ok(()),
        };
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_double_crlf(&buf) {
            header_end = pos;
            break;
        }
        if buf.len() > 64 * 1024 {
            return write_status(&mut stream, 413, "Request Too Large");
        }
    }

    let head_str = match std::str::from_utf8(&buf[..header_end]) {
        Ok(s) => s,
        Err(_) => return write_status(&mut stream, 400, "Bad Request"),
    };
    let mut lines = head_str.lines();
    let request_line = match lines.next() {
        Some(l) => l,
        None => return Ok(()),
    };
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return write_status(&mut stream, 400, "Bad Request");
    }
    let method = parts[0];
    let path = parts[1];

    let mut content_length = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }

    let body_start = header_end + 4;
    let mut body: Vec<u8> = Vec::new();
    if content_length > 0 {
        if body_start < buf.len() {
            body.extend_from_slice(&buf[body_start..]);
        }
        while body.len() < content_length {
            let need = content_length - body.len();
            let mut chunk = vec![0u8; need.min(8192)];
            match stream.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => body.extend_from_slice(&chunk[..n]),
                Err(_) => break,
            }
        }
    }

    match method {
        "OPTIONS" => write_cors_preflight(&mut stream),
        "GET" | "HEAD" => serve_static(&mut stream, path, root),
        "POST" if path.starts_with("/api/") => proxy_ollama(&mut stream, &body, path),
        _ => write_status(&mut stream, 405, "Method Not Allowed"),
    }
}

pub(crate) fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    for i in 0..=buf.len() - 4 {
        if &buf[i..i + 4] == b"\r\n\r\n" {
            return Some(i);
        }
    }
    None
}

pub(crate) fn serve_static(
    stream: &mut std::net::TcpStream,
    path: &str,
    root: &str,
) -> std::io::Result<()> {
    let clean = path.split('?').next().unwrap_or(path);
    let clean = if clean == "/" { "/graph.html" } else { clean };
    // path-traversal guard
    if clean.contains("..") || !clean.starts_with('/') {
        return write_status(stream, 400, "Bad Request");
    }
    let full = format!("{}{}", root, clean);
    match std::fs::read(&full) {
        Ok(data) => write_response(stream, 200, "OK", content_type_for(clean), &data),
        Err(_) => write_status(stream, 404, "Not Found"),
    }
}

pub(crate) fn content_type_for(path: &str) -> &'static str {
    if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if path.ends_with(".json") {
        "application/json; charset=utf-8"
    } else if path.ends_with(".js") {
        "application/javascript; charset=utf-8"
    } else if path.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".png") {
        "image/png"
    } else if path.ends_with(".txt") {
        "text/plain; charset=utf-8"
    } else {
        "application/octet-stream"
    }
}

pub(crate) fn proxy_ollama(
    stream: &mut std::net::TcpStream,
    body: &[u8],
    path: &str,
) -> std::io::Result<()> {
    let url = format!("http://localhost:11434{}", path);
    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .timeout(std::time::Duration::from_secs(600))
        .send_bytes(body);
    match resp {
        Ok(r) => {
            let status = r.status();
            let body = r.into_string().unwrap_or_default();
            write_response(
                stream,
                status,
                "OK",
                "application/json; charset=utf-8",
                body.as_bytes(),
            )
        }
        Err(ureq::Error::Status(code, r)) => {
            let body = r.into_string().unwrap_or_default();
            write_response(
                stream,
                code,
                "OLLAMA ERROR",
                "application/json; charset=utf-8",
                body.as_bytes(),
            )
        }
        Err(e) => {
            let msg = format!("{{\"error\":\"failed to reach ollama: {}\"}}", e);
            write_response(
                stream,
                502,
                "Bad Gateway",
                "application/json; charset=utf-8",
                msg.as_bytes(),
            )
        }
    }
}

pub(crate) fn write_response(
    stream: &mut std::net::TcpStream,
    code: u16,
    status: &str,
    ct: &str,
    body: &[u8],
) -> std::io::Result<()> {
    use std::io::Write;
    let header = format!(
        "HTTP/1.1 {} {}\r\n\
         Content-Type: {}\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Cache-Control: no-cache\r\n\
         Connection: close\r\n\r\n",
        code,
        status,
        ct,
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

pub(crate) fn write_status(
    stream: &mut std::net::TcpStream,
    code: u16,
    status: &str,
) -> std::io::Result<()> {
    let body = format!("{} {}", code, status);
    write_response(stream, code, status, "text/plain; charset=utf-8", body.as_bytes())
}

pub(crate) fn write_cors_preflight(stream: &mut std::net::TcpStream) -> std::io::Result<()> {
    use std::io::Write;
    let header = "HTTP/1.1 204 No Content\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Connection: close\r\n\r\n";
    stream.write_all(header.as_bytes())?;
    stream.flush()
}

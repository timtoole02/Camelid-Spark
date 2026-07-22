// Logging reverse proxy: 127.0.0.1:8182 -> 127.0.0.1:8181 (Camelid).
// Tees BOTH the inbound request body and the upstream response body to a log,
// so a client session proves (a) what the client sent (stream_options) and
// (b) what came back on the wire (the terminal usage chunk). SSE-safe: the
// response is piped through unchanged while a copy is appended to the log.
// No deps; Node >= 18.
import http from "node:http";
import fs from "node:fs";

const PORT = Number(process.env.PROXY_PORT || 8182);
const UP_HOST = process.env.UP_HOST || "127.0.0.1";
const UP_PORT = Number(process.env.UP_PORT || 8181);
const LOG = process.env.PROXY_LOG || "proxy.log";

function log(line) {
  fs.appendFileSync(LOG, line + "\n");
}

const server = http.createServer((req, res) => {
  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", () => {
    const body = Buffer.concat(chunks);
    const ts = new Date().toISOString();
    log(`=== ${ts} ${req.method} ${req.url} (req body ${body.length}B) ===`);
    if (body.length) log("REQ_BODY " + body.toString("utf8"));
    const headers = { ...req.headers, host: `${UP_HOST}:${UP_PORT}` };
    const up = http.request(
      { host: UP_HOST, port: UP_PORT, path: req.url, method: req.method, headers },
      (upRes) => {
        log(`RESP ${upRes.statusCode} ${req.method} ${req.url}`);
        res.writeHead(upRes.statusCode, upRes.headers);
        const respChunks = [];
        upRes.on("data", (c) => {
          respChunks.push(c);
          res.write(c);
        });
        upRes.on("end", () => {
          const rb = Buffer.concat(respChunks).toString("utf8");
          // Only tee chat-completions traffic to keep the log focused.
          if (req.url.includes("chat/completions")) {
            log("RESP_BODY_BEGIN");
            log(rb);
            log("RESP_BODY_END");
          }
          res.end();
        });
      }
    );
    up.on("error", (e) => {
      log("UPSTREAM_ERR " + e.message);
      if (!res.headersSent) res.writeHead(502, { "content-type": "text/plain" });
      res.end("proxy upstream error: " + e.message);
    });
    if (body.length) up.write(body);
    up.end();
  });
});

server.listen(PORT, "127.0.0.1", () =>
  log(`[proxy] listening 127.0.0.1:${PORT} -> ${UP_HOST}:${UP_PORT}`)
);

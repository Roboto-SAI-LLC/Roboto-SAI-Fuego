import http from 'http';

interface MockState {
  modelsStatus: number;
  healthStatus: number;
  completionText: string;
}

const state: MockState = {
  modelsStatus: 200,
  healthStatus: 200,
  completionText: '\n// mock completion\n'
};

const server = http.createServer((req, res) => {
  const { method, url } = req;

  if (method === 'GET' && url === '/v1/models') {
    res.writeHead(state.modelsStatus, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ data: [{ id: 'local', object: 'model' }] }));
    return;
  }

  if (method === 'GET' && url === '/health') {
    res.writeHead(state.healthStatus, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok' }));
    return;
  }

  if (method === 'POST' && url === '/v1/completions') {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk.toString();
    });
    req.on('end', () => {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          id: 'cmpl-mock',
          object: 'text_completion',
          model: 'local',
          choices: [{ text: state.completionText }]
        })
      );
    });
    return;
  }

  if (method === 'POST' && url === '/__mock__/config') {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk.toString();
    });
    req.on('end', () => {
      const payload = body ? (JSON.parse(body) as Partial<MockState>) : {};
      if (typeof payload.modelsStatus === 'number') {
        state.modelsStatus = payload.modelsStatus;
      }
      if (typeof payload.healthStatus === 'number') {
        state.healthStatus = payload.healthStatus;
      }
      if (typeof payload.completionText === 'string') {
        state.completionText = payload.completionText;
      }

      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(state));
    });
    return;
  }

  res.writeHead(404, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ error: 'Not found' }));
});

const port = Number(process.env.MOCK_PORT ?? '8787');
server.listen(port, '127.0.0.1', () => {
  process.stdout.write(`mock-server listening on 127.0.0.1:${port}\n`);
});

process.on('SIGTERM', () => {
  server.close(() => process.exit(0));
});

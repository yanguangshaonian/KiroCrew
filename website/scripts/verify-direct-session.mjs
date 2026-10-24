import assert from 'node:assert/strict'

const base = process.env.KIROCREW_TEST_URL || 'http://127.0.0.1:5476'
const origin = new URL(base).origin
const marker = `DIRECT_CONTEXT_${Date.now()}`
let slot

async function request(path, body, method = 'POST') {
  const response = await fetch(`${base}${path}`, {
    method,
    headers: { Origin: origin, 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(20000),
  })
  const text = await response.text()
  assert(response.ok, `${path}: HTTP ${response.status}: ${text.slice(0, 500)}`)
  return text ? JSON.parse(text) : {}
}

async function chat(message) {
  const response = await fetch(`${base}/api/chat`, {
    method: 'POST',
    headers: { Origin: origin, 'Content-Type': 'application/json' },
    body: JSON.stringify({ slot, message }),
    signal: AbortSignal.timeout(90000),
  })
  assert(response.ok, `聊天请求失败: HTTP ${response.status}`)
  const stream = await response.text()
  assert(stream.includes('data: [DONE]'), '聊天流缺少完成标记.')
  const events = stream.split('\n').filter(line => line.startsWith('data: {')).map(line => JSON.parse(line.slice(6)))
  const errors = events.filter(event => event.type === 'error')
  assert.equal(errors.length, 0, JSON.stringify(errors))
  return events.filter(event => event.type === 'assistant').map(event => event.content).join('\n')
}

try {
  const created = await request('/api/chat/slots', { agent: 'kirocrew-lite', model: 'gpt-6.1-sol', memory_mode: 'temporary' })
  slot = created.key || created.slot
  assert(slot, `创建会话没有返回 key: ${JSON.stringify(created)}`)
  await chat(`记住标记 ${marker}, 只回复已记住, 不使用工具.`)
  assert((await chat('返回我上一条消息中的完整标记, 不使用工具.')).includes(marker), '第二轮未保留上下文.')
  await request(`/api/chat/slots/${encodeURIComponent(slot)}/model`, { model: 'grok-4.7' })
  assert((await chat('返回我先前提供的完整标记, 不使用工具.')).includes(marker), '切换模型后丢失上下文.')
  const socketUrl = new URL('/api/ws', base)
  socketUrl.protocol = socketUrl.protocol === 'https:' ? 'wss:' : 'ws:'
  await new Promise((resolve, reject) => {
    const socket = new WebSocket(socketUrl)
    const timer = setTimeout(() => { socket.close(); reject(new Error('WebSocket 验证超时.')) }, 15000)
    socket.addEventListener('message', () => { clearTimeout(timer); socket.close(); resolve() }, { once: true })
    socket.addEventListener('error', () => { clearTimeout(timer); socket.close(); reject(new Error('WebSocket 连接失败.')) }, { once: true })
  })
  console.log('两轮上下文、跨模型切换与免 token WebSocket 检查通过.')
} finally {
  if (slot) await request(`/api/chat/slots/${encodeURIComponent(slot)}`, undefined, 'DELETE')
}
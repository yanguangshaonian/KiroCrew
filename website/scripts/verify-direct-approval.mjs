import assert from 'node:assert/strict'

const base = process.env.KIROCREW_TEST_URL || 'http://127.0.0.1:5476'
const headers = { Origin: new URL(base).origin, 'Content-Type': 'application/json' }
let slot
let approved = false
let completed = false

async function request(path, body, method = 'POST') {
  const response = await fetch(`${base}${path}`, {
    method, headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(15000),
  })
  const text = await response.text()
  assert(response.ok, `请求失败: ${path}, HTTP ${response.status}: ${text.slice(0, 300)}`)
  return text ? JSON.parse(text) : {}
}

try {
  const created = await request('/api/chat/slots', { agent: 'kirocrew', memory_mode: 'temporary' })
  slot = created.key || created.slot
  assert(slot, '创建会话未返回标识.')
  const response = await fetch(`${base}/api/chat`, {
    method: 'POST', headers,
    body: JSON.stringify({ slot, message: '请实际调用 kirocrew-cron 的 cron_list 查询当前任务. 不要创建或修改任务, 简短报告结果.' }),
    signal: AbortSignal.timeout(90000),
  })
  assert(response.ok, `聊天失败: HTTP ${response.status}`)
  const decoder = new TextDecoder()
  let pending = ''
  let done = false
  for await (const chunk of response.body) {
    pending += decoder.decode(chunk, { stream: true })
    const lines = pending.split('\n')
    pending = lines.pop()
    for (const line of lines) {
      if (line === 'data: [DONE]') done = true
      if (!line.startsWith('data: {')) continue
      const event = JSON.parse(line.slice(6))
      assert.notEqual(event.type, 'error', event.content)
      if (event.type === 'permission' && event.meta?.approval_id) {
        assert.equal(event.meta.tool_name, 'cron_list', '拒绝批准非查询工具.')
        await request(`/api/chat/slots/${encodeURIComponent(slot)}/approve`, {
          request_id: event.meta.approval_id, action: 'approved',
        })
        approved = true
      }
      if (event.type === 'tool' && /cron_list/.test(event.content) && /completed|✅/.test(JSON.stringify(event))) completed = true
    }
  }
  assert(done, '聊天流缺少完成标记.')
  assert(approved, '未观察到审批卡片.')
  assert(completed, '未观察到成功工具结果.')
  console.log('真实聊天审批、cron_list 执行与模型续答检查通过.')
} finally {
  if (slot) await request(`/api/chat/slots/${encodeURIComponent(slot)}`, undefined, 'DELETE')
}   
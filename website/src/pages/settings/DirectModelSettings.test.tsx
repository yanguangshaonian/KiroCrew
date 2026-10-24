import React from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { DirectModelSettings } from './DirectModelSettings'

vi.mock('../../components/ErrorNotice', () => ({
  default: ({ message }: { message: string }) => message ? <div role="alert">{message}</div> : null,
}))

const settings = {
  url: 'http://localhost:20016/v1/chat/completions',
  model: 'test-model',
  models: [{ id: 'test-model' }],
  has_api_key: true,
}
let fetchMock: ReturnType<typeof vi.fn>
let queryClient: QueryClient

beforeEach(() => {
  vi.spyOn(AbortSignal, 'timeout').mockImplementation(() => new AbortController().signal)
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  fetchMock = vi.fn().mockImplementation(async (_url, options) => new Response(
    JSON.stringify(options?.method === 'PUT' ? { ok: true } : settings),
    { status: 200, headers: { 'Content-Type': 'application/json' } },
  ))
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

async function showForm() {
  render(<QueryClientProvider client={queryClient}><DirectModelSettings /></QueryClientProvider>)
  await screen.findByLabelText('模型 URL', { exact: true })
}

describe('直接模型设置', () => {
  it('保存空 key 并为请求设置截止信号', async () => {
    await showForm()
    expect(screen.getByLabelText('API key')).toHaveValue('')
    fireEvent.click(screen.getByRole('button', { name: '保存模型连接' }))
    await screen.findByText('已保存.')
    const request = fetchMock.mock.calls.find(([, options]) => options?.method === 'PUT')
    expect(request).toBeDefined()
    expect(JSON.parse(request![1].body)).toEqual({ ...settings, api_key: '', has_api_key: undefined })
    expect(request![1].signal).toBeInstanceOf(AbortSignal)
  })

  it('模型列表不是数组时不提交请求', async () => {
    await showForm()
    fireEvent.change(screen.getByLabelText('模型列表'), { target: { value: '{}' } })
    fireEvent.click(screen.getByRole('button', { name: '保存模型连接' }))
    await screen.findByRole('alert')
    expect(screen.getByRole('alert')).toHaveTextContent('模型列表必须是 JSON 数组.')
    expect(fetchMock.mock.calls.filter(([, options]) => options?.method === 'PUT')).toHaveLength(0)
  })

  it('保存失败时显示错误且保留草稿', async () => {
    await showForm()
    fetchMock.mockImplementation(async () => new Response(JSON.stringify({ error: '连接配置无效.' }), { status: 400 }))
    fireEvent.change(screen.getByLabelText('模型 URL', { exact: true }), { target: { value: 'http://localhost/changed' } })
    fireEvent.click(screen.getByRole('button', { name: '保存模型连接' }))
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('连接配置无效.'))
    expect(screen.getByLabelText('模型 URL', { exact: true })).toHaveValue('http://localhost/changed')
    expect(screen.queryByText('已保存.')).toBeNull()
  })
})
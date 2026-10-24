import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Save } from 'lucide-react'
import { Input, PanelSectionHeader, Skeleton } from '../../components/ui'
import ErrorNotice from '../../components/ErrorNotice'

type ModelSettings = {
  url: string
  model: string
  models: Array<Record<string, unknown>>
  has_api_key: boolean
}

async function responseJson(response: Response) {
  const body = await response.json()
  if (!response.ok) throw new Error(body.error || '模型设置请求失败.')
  return body
}

function ModelForm({ settings }: { settings: ModelSettings }) {
  const queryClient = useQueryClient()
  const [url, setUrl] = useState(settings.url || '')
  const [key, setKey] = useState('')
  const [model, setModel] = useState(settings.model || '')
  const [models, setModels] = useState(JSON.stringify(settings.models || [], null, 2))
  const [saved, setSaved] = useState(false)
  const mutation = useMutation({
    mutationFn: async () => {
      const catalog: unknown = JSON.parse(models)
      if (!Array.isArray(catalog)) throw new Error('模型列表必须是 JSON 数组.')
      return fetch('/api/config/direct-model', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(15000),
        body: JSON.stringify({ url, api_key: key, model, models: catalog }),
      }).then(responseJson)
    },
    onSuccess: () => {
      setKey('')
      setSaved(true)
      void queryClient.invalidateQueries({ queryKey: ['directModelSettings'] })
      void queryClient.invalidateQueries({ queryKey: ['kirocrewConfig'] })
      void queryClient.invalidateQueries({ predicate: query => query.queryKey.some(part => String(part).toLowerCase().includes('model')) })
    },
  })
  const changed = () => setSaved(false)
  return <form className="mb-8 space-y-4" onSubmit={event => { event.preventDefault(); mutation.mutate() }}>
    <PanelSectionHeader label="模型连接" />
    <label className="block text-sm">模型 URL
      <Input required type="url" value={url} onChange={event => { setUrl(event.target.value); changed() }} className="mt-2 w-full" />
    </label>
    <label className="block text-sm">API key
      <Input type="password" autoComplete="new-password" value={key} onChange={event => { setKey(event.target.value); changed() }} placeholder={settings.has_api_key ? '已配置, 留空保留' : ''} className="mt-2 w-full" />
    </label>
    <label className="block text-sm">默认模型
      <Input required value={model} onChange={event => { setModel(event.target.value); changed() }} className="mt-2 w-full" />
    </label>
    <label className="block text-sm">模型列表
      <textarea required value={models} onChange={event => { setModels(event.target.value); changed() }} rows={12} spellCheck={false} className="mt-2 block w-full resize-y rounded border border-[var(--border)] bg-[var(--bg)] p-3 font-mono text-sm text-[var(--text)]" />
    </label>
    <ErrorNotice message={mutation.error?.message || ''} />
    <div className="flex items-center gap-3">
      <span role="button" tabIndex={mutation.isPending ? -1 : 0} aria-disabled={mutation.isPending} aria-label="保存模型连接" title="保存模型连接" className="inline-flex cursor-pointer items-center gap-2 rounded border border-[var(--border)] px-3 py-2 text-sm" onClick={() => { if (!mutation.isPending) mutation.mutate() }} onKeyDown={event => { if ((event.key === 'Enter' || event.key === ' ') && !mutation.isPending) { event.preventDefault(); mutation.mutate() } }}>
        <Save size={16} />{mutation.isPending ? '保存中...' : '保存'}
      </span>
      {saved && <span role="status" className="text-sm">已保存.</span>}
    </div>
  </form>
}

export function DirectModelSettings() {
  const query = useQuery<ModelSettings>({ queryKey: ['directModelSettings'], queryFn: () => fetch('/api/config/direct-model', { signal: AbortSignal.timeout(15000) }).then(responseJson) })
  if (query.isPending) return <Skeleton className="mb-8 h-40" />
  if (query.isError) return <ErrorNotice message={query.error.message} />
  return <ModelForm settings={query.data} />
}
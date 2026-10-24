import type { ProviderAdapter, ProviderId } from './types'
import { DirectAdapter } from './adapters/direct'

const adapter = new DirectAdapter()
export function getAdapter(_id?: ProviderId): ProviderAdapter {
  return adapter
}

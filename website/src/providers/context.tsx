import { createContext, useContext, type ReactNode } from 'react'
import { getAdapter } from './registry'
import type { ProviderAdapter } from './types'

const directAdapter = getAdapter()

const ProviderContext = createContext<ProviderAdapter>(directAdapter)

export function ProviderProvider({ children }: { children: ReactNode }) {
  return <ProviderContext.Provider value={directAdapter}>{children}</ProviderContext.Provider>
}

export function useProvider(): ProviderAdapter {
  return useContext(ProviderContext)
}

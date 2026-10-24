import { chromium } from 'playwright-core'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'

const base = process.env.KIROCREW_TEST_URL || 'http://127.0.0.1:5476'
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH || `${process.env.HOME}/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome`,
  headless: true,
  timeout: 15000,
  args: ['--no-sandbox'],
})
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
  page.setDefaultTimeout(15000)
  page.setDefaultNavigationTimeout(20000)
  await page.goto(`${base}/settings/chat/models`, { waitUntil: 'domcontentloaded' })
  const privacy = page.getByRole('dialog', { name: '隐私', exact: true })
  if (await privacy.isVisible()) await privacy.getByRole('button', { name: '继续', exact: true }).click()
  const skip = page.getByRole('button', { name: '跳过所有设置与引导' })
  if (await skip.isVisible()) await skip.click()
  await page.getByLabel('模型 URL', { exact: true }).waitFor({ timeout: 30000 })
  assert.match(await page.getByLabel('模型 URL', { exact: true }).inputValue(), /chat\/completions/)
  assert.equal(await page.getByLabel('默认模型', { exact: true }).first().inputValue(), 'gpt-6.1-sol')
  await page.getByRole('button', { name: '保存模型连接', exact: true }).click()
  await page.getByRole('status').filter({ hasText: '已保存.' }).waitFor()
  await fs.mkdir('temp-screenshots', { recursive: true })
  await page.screenshot({ path: 'temp-screenshots/direct-model-desktop.png', fullPage: true })
  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({ path: 'temp-screenshots/direct-model-mobile.png', fullPage: true })
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1))
  console.log('模型设置加载、保存与移动端宽度检查通过.')
} finally {
  await browser.close()
}
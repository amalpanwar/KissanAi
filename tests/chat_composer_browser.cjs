// Browser integration test against the actual Streamlit component handshake.
const { chromium } = require('playwright');
const { spawn } = require('node:child_process');
const net = require('node:net');
const assert = require('node:assert/strict');

async function freePort() {
  const server = net.createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

(async () => {
  const port = await freePort();
  const python = process.env.PYTHON || 'python';
  const server = spawn(python, ['-m', 'streamlit', 'run', 'tests/fixtures/chat_composer_app.py',
    '--server.address=127.0.0.1', `--server.port=${port}`, '--server.headless=true', '--browser.gatherUsageStats=false']);
  let logs = '';
  server.stdout.on('data', chunk => { logs += chunk; });
  server.stderr.on('data', chunk => { logs += chunk; });
  let browser;
  try {
    const url = `http://127.0.0.1:${port}`;
    const deadline = Date.now() + 30000;
    while (true) {
      try { if ((await fetch(`${url}/_stcore/health`)).ok) break; } catch {}
      if (Date.now() > deadline) throw new Error(`Streamlit failed to start: ${logs}`);
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined });
    const page = await browser.newPage({ viewport: { width: 1000, height: 800 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(url);
    const frame = page.frameLocator('iframe[title="app.chat_controls.kisaan_chat_composer"]');
    const input = frame.getByRole('combobox');
    await input.waitFor();
    const options = frame.getByRole('option');
    async function assertComposerBelowReplies() {
      const order = await page.evaluate(() => {
        const replies = [...document.querySelectorAll('[data-testid="stChatMessage"]')];
        const composer = document.querySelector('iframe[title="app.chat_controls.kisaan_chat_composer"]');
        return replies.length > 0 && replies.every(reply =>
          Boolean(reply.compareDocumentPosition(composer) & Node.DOCUMENT_POSITION_FOLLOWING) &&
          reply.getBoundingClientRect().bottom <= composer.getBoundingClientRect().top);
      });
      assert.equal(order, true, 'All replies must render above the composer');
    }

    await input.fill('wea');
    await options.first().waitFor();
    assert.match(await options.first().innerText(), /मौसम/);
    assert.equal(await options.count(), 1);
    // Nothing is submitted while typing or selecting a suggestion.
    assert.equal((await page.getByTestId('stJson').innerText()).trim(), '[]');
    await options.first().click();
    assert.match(await input.inputValue(), /मेरे क्षेत्र/);
    assert.equal((await page.getByTestId('stJson').innerText()).trim(), '[]');
    await input.fill('Doghat Rural में आज मौसम कैसा है?');
    await input.press('Enter');
    await page.getByTestId('stJson').filter({ hasText: 'Doghat Rural' }).waitFor();
    assert.equal(await input.inputValue(), '');
    await assertComposerBelowReplies();
    await page.getByRole('button', { name: 'Rerun unrelated control' }).click();
    await input.fill('pesti');
    await options.first().waitFor();
    assert.match(await options.first().innerText(), /कीटनाशक/);
    await assertComposerBelowReplies();
    await input.press('ArrowDown');
    await input.press('Enter');
    assert.match(await input.inputValue(), /कीटनाशक/);
    await frame.getByRole('button', { name: 'सवाल भेजें' }).click();
    await page.getByTestId('stJson').filter({ hasText: 'दीमक' }).waitFor();
    await assertComposerBelowReplies();
    let history = await page.getByTestId('stJson').innerText();
    assert.equal((history.match(/Doghat Rural/g) || []).length, 1);
    assert.equal((history.match(/दीमक/g) || []).length, 1);
    await input.fill('price');
    await options.first().waitFor();
    assert.match(await options.first().innerText(), /मंडी भाव/);
    await input.press('Escape');
    assert.equal(await options.count(), 0);
    await input.fill('kheti');
    await options.first().waitFor();
    assert.match(await options.first().innerText(), /खेती/);
    await input.press('Shift+Enter');
    assert.ok((await input.inputValue()).includes('\n'));
    // Mobile width: no horizontal overflow, same input and suggestion selection.
    await page.setViewportSize({ width: 390, height: 844 });
    await input.fill('मौसम');
    await options.first().waitFor();
    const overflow = await frame.locator('body').evaluate(el => el.scrollWidth > document.documentElement.clientWidth);
    assert.equal(overflow, false);
    await assertComposerBelowReplies();
    if (process.env.COMPOSER_SCREENSHOT) await page.screenshot({ path: process.env.COMPOSER_SCREENSHOT, fullPage: true });
    assert.deepEqual(errors, []);
    console.log('Chat composer browser tests passed: typing, click/keyboard suggestions, edited send, no duplicate rerun, Escape, multiline and mobile.');
  } finally {
    if (browser) await browser.close();
    server.kill('SIGTERM');
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

import fs from 'fs';
const f = new URL('./FIXTURES_tokenizer_adversarial.json', import.meta.url);
const cases = JSON.parse(fs.readFileSync(f, 'utf8'));
cases[10] = 'café naïve résumé (NFD combining marks)'.normalize('NFD');
cases[11] = 'café naïve résumé (NFC precomposed)'.normalize('NFC');
cases[12] = 'Việt Nam (NFC) vs ' + 'Việt Nam (NFD)'.normalize('NFD');
// ascii-escape all non-ASCII so the byte content is unambiguous in review
const esc = (s) =>
  JSON.stringify(s).replace(/[-￿]/g, (c) => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0'));
fs.writeFileSync(f, '[\n' + cases.map((c) => '  ' + esc(c)).join(',\n') + '\n]\n');
const back = JSON.parse(fs.readFileSync(f, 'utf8'));
const hasCombining = (s) => /[̀-ͯ]/.test(s);
console.log(
  'entries:', back.length,
  '| case10 NFD:', hasCombining(back[10]) ? 'OK' : 'FAIL',
  '| case11 NFC:', !hasCombining(back[11]) ? 'OK' : 'FAIL',
  '| case12 mixed:', /̣/.test(back[12]) ? 'OK' : 'FAIL'
);

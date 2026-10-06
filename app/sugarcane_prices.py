"""Dated official cane-price notices, distinct from Agmarknet transactions.

Update these reviewed notices when a new official season notification is verified.
No baseline estimate or undated scraped price is presented as a current rate.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from app.agent_system import AgentResult
from app.location_selection import qualified_place

VERIFIED_ON = '2026-10-06'
UP_SOURCE = 'https://information.up.gov.in/admin/UploadDocument/OtherPress/compressed_15112025081854PN-CM-Cabinet%20Decisions-14%20November%2C%202025.pdf'
FRP_SOURCE = 'https://www.pib.gov.in/PressReleasePage.aspx?PRID=2258142'


def sugarcane_price_result(location, today=None):
    today = today or datetime.now(ZoneInfo('Asia/Kolkata')).date()
    start = today.year if today.month >= 10 else today.year - 1
    current_season = f'{start}-{str(start + 1)[-2:]}'
    is_up = str(location.get('state') or '').strip().casefold() in {'uttar pradesh', 'उत्तर प्रदेश'}
    lines = [f"**गन्ने का सरकारी खरीद मूल्य — {qualified_place(location) or 'भारत'}**",
             'ये चीनी मिलों की खरीद के लिए सरकारी दरें हैं; गांव की आज की मंडी बोली नहीं।']
    references = []
    notices = []
    if is_up and today.isoformat() >= '2025-11-14':
        label = 'इस सत्र का रिकॉर्ड' if current_season == '2025-26' else 'पुराने सत्र का रिकॉर्ड; आज की दर न मानें'
        lines += [f'उत्तर प्रदेश राज्य परामर्शित मूल्य (SAP), पेराई सत्र 2025–26 — {label}:',
                  '- अगेती प्रजाति: ₹400/क्विंटल', '- सामान्य प्रजाति: ₹390/क्विंटल',
                  '- अनुपयुक्त प्रजाति: ₹355/क्विंटल', f'[उत्तर प्रदेश सरकार, 14-11-2025]({UP_SOURCE})']
        references.append(UP_SOURCE)
        notices.append({'kind': 'SAP', 'state': 'Uttar Pradesh', 'season': '2025-26',
                        'prices_per_quintal': {'early': 400, 'general': 390, 'unsuitable': 355}, 'source': UP_SOURCE})
        if current_season != '2025-26':
            lines.append(f'सत्र {current_season} की उत्तर प्रदेश SAP अधिसूचना इस ऐप के सत्यापित रिकॉर्ड में उपलब्ध नहीं है।')
    if today.isoformat() >= '2026-05-05':
        label = ('इस सत्र की दर' if current_season == '2026-27' else
                 'आगामी सत्र की घोषित दर' if today.isoformat() < '2026-10-01' else 'पुराने सत्र की दर; आज की दर न मानें')
        lines += [f'केंद्र का उचित एवं लाभकारी मूल्य (FRP), सत्र 2026–27 — {label}:',
                  '- ₹365/क्विंटल, 10.25% चीनी रिकवरी पर; लागू अवधि 01-10-2026 से 30-09-2027।',
                  '- रिकवरी दर के अनुसार मूल्य बदलता है; इसे उत्तर प्रदेश की SAP दर न मानें।',
                  f'[भारत सरकार, 05-05-2026]({FRP_SOURCE})']
        references.append(FRP_SOURCE)
        notices.append({'kind': 'FRP', 'season': '2026-27', 'price_per_quintal': 365,
                        'basic_recovery_percent': 10.25, 'source': FRP_SOURCE})
    lines += [f'स्रोतों का अंतिम सत्यापन: {VERIFIED_ON}।',
              'अपनी किस्म और चालू पेराई सत्र की देय दर चीनी मिल/गन्ना समिति की पर्ची से मिलाएं।']
    if not notices:
        return AgentResult('इस तारीख के लिए सत्यापित गन्ना खरीद मूल्य उपलब्ध नहीं है।', 'unavailable')
    current = any(n['season'] == current_season for n in notices)
    status = 'partial' if is_up and current_season != '2025-26' and current else 'ok' if current else 'stale'
    return AgentResult('\n\n'.join(lines), status, references,
                       {'location': location, 'price_basis': 'official_purchase_notice',
                        'notices': notices, 'verified_on': VERIFIED_ON, 'current_season': current_season})

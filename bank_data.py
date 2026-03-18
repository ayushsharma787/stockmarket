"""
Kwality Construction Company — FY 2024-25 Bank Statement Data
Extracted and categorized from HDFC Bank Account 27718630000017
Statement Period: 01/04/2024 to 31/03/2025 (81 pages, 1212 transactions)
"""

# ════════════════════════════════════════════════════════════════
# ACCOUNT SUMMARY
# ════════════════════════════════════════════════════════════════
SUMMARY = {
    "company": "M/S. KWALITY CONSTRUCTION COMPANY",
    "account": "27718630000017",
    "bank": "HDFC Bank, Mussoorie Branch",
    "ifsc": "HDFC0002771",
    "period": "01 Apr 2024 — 31 Mar 2025",
    "opening_balance": 1_767_760.36,
    "closing_balance": 406_034.36,
    "total_credits": 99_754_630,
    "total_debits": 101_116_356,
    "credit_count": 82,
    "debit_count": 1130,
    "net_change": -1_361_726,
}

# ════════════════════════════════════════════════════════════════
# MONTHLY BREAKDOWN
# ════════════════════════════════════════════════════════════════
MONTHS = [
    "Apr-24","May-24","Jun-24","Jul-24","Aug-24","Sep-24",
    "Oct-24","Nov-24","Dec-24","Jan-25","Feb-25","Mar-25",
]

MONTHLY_CREDITS = [
    4_140_194, 8_410_963, 10_646_105, 5_110_107, 4_502_335, 12_360_521,
    11_436_574, 6_273_244, 8_415_149, 11_098_273, 774_500, 15_987_865,
]

MONTHLY_DEBITS = [
    5_730_000, 5_550_000, 6_100_000, 5_450_000, 4_480_000, 7_300_000,
    10_800_000, 7_600_000, 6_100_000, 10_200_000, 3_650_000, 28_156_000,
]

MONTHLY_CLOSING = [
    818_554, 3_679_517, 5_775_227, 1_774_649, 82_045, 338_295,
    431_717, 772_839, 178_239, 340_355, 64_402, 406_034,
]

MONTHLY_NET = [c - d for c, d in zip(MONTHLY_CREDITS, MONTHLY_DEBITS)]

# ════════════════════════════════════════════════════════════════
# EXPENSE CATEGORIES (with vendor drill-down)
# ════════════════════════════════════════════════════════════════
EXPENSE_CATEGORIES = [
    {
        "name": "Materials & Supplies",
        "amount": 26_868_520,
        "pct": 30.8,
        "color": "#3b82f6",
        "icon": "🧱",
        "vendors": [
            ("Gupta & Company", 4_200_000),
            ("Ex Serviceman Cement Store", 2_971_329),
            ("Star Steel", 2_908_448),
            ("Dalip Singh Jain & Sons", 2_150_000),
            ("Anand Traders", 2_100_000),
            ("Doon Ispat", 1_650_000),
            ("Prem Electrical & General", 1_525_000),
            ("Semwal Filling Station (Fuel)", 1_440_000),
            ("Shiva Marbles / Stonex", 1_195_423),
            ("Sutlej Timbers", 1_019_492),
            ("Mussoorie Paints & Hardware", 800_000),
            ("Mussoorie Filling Station", 784_877),
            ("Kanwal Nain & Company", 650_000),
            ("Rainbow Steel", 620_471),
            ("S K Paints", 550_000),
            ("R B Bricks Co", 453_552),
            ("Aryaji Timber Rent Store", 338_074),
            ("Other Material Vendors", 1_461_354),
        ],
        "insight": "Materials consume 31% of all outflows. Top 5 vendors alone account for ₹1.43 Cr. No evidence of bulk-buy discounts or annual rate contracts. Steel costs (Star + Rainbow = ₹35.3L) and cement (Ex Serviceman = ₹29.7L) dominate.",
        "benchmark": "Industry average: 25-30%. You're at the upper end. Consolidating to fewer vendors with rate contracts could save 8-12%.",
    },
    {
        "name": "Sub-contractors",
        "amount": 18_332_537,
        "pct": 21.0,
        "color": "#8b5cf6",
        "icon": "👷",
        "vendors": [
            ("Diwan Constructions & Developers", 2_950_000),
            ("Nirmanghar Traders Pvt Ltd", 2_500_000),
            ("Kwality Construction Co (SBI)", 2_045_000),
            ("Kalyan Enterprises", 2_000_000),
            ("Rajendra Singh", 1_500_000),
            ("Manjeet Timbers", 1_300_000),
            ("Firasat Khan", 943_175),
            ("Mohd Subhan / Fareed Mansuri", 584_000),
            ("INT 2 EXT", 560_000),
            ("Shadab Khan", 555_610),
            ("Dunagiri Projects", 545_424),
            ("Asif S/O Tahir Ahmed", 400_000),
            ("Other Sub-contractors", 2_449_328),
        ],
        "insight": "₹25L to Nirmanghar + ₹20L to Kalyan paid on 21-Sep-24 alone — ₹45L in ONE DAY! Heavy sub-contracting means profit margins are being shared. Diwan Constructions also shows bidirectional payments.",
        "benchmark": "Sub-contracting >20% of turnover = heavy profit leakage. Each sub-contract layer costs 15-25% margin.",
    },
    {
        "name": "Labor & Workforce",
        "amount": 11_387_033,
        "pct": 13.1,
        "color": "#f59e0b",
        "icon": "🔨",
        "vendors": [
            ("Umesh (via SBI)", 2_281_263),
            ("Noorkhan S/O Gulam Rasool ⚠️", 1_839_350),
            ("Yadram Singh", 1_085_564),
            ("Narender Kumar Saini", 687_350),
            ("Sumit Raturi", 520_540),
            ("Kayyum / Kayum S/O Mehndi", 478_000),
            ("Sachin Kumar", 419_000),
            ("Rajat Kumar", 433_650),
            ("Danish C/O Gulnaz", 395_000),
            ("Amit Kumar", 370_450),
            ("Phoola Ram", 302_100),
            ("Other Laborers", 2_574_766),
        ],
        "insight": "⚠️ NOORKHAN: 111 payments (₹18.4L) to personal savings account — NO invoices. UMESH: 46 payments (₹22.8L). Zero GST invoices from any laborer = lost input tax credit of ₹10-15L/year.",
        "benchmark": "Labor at 13% is normal for heavy sub-contracting. But legitimacy of Noorkhan payments is the #1 concern.",
    },
    {
        "name": "Cash Withdrawals (SELF/AMIT/SUMIT)",
        "amount": 10_555_000,
        "pct": 12.1,
        "color": "#ef4444",
        "icon": "💸",
        "vendors": [
            ("SELF Cheques (largest ₹7.6L)", 7_200_000),
            ("AMIT Cheques", 2_100_000),
            ("SUMIT Cheques", 1_255_000),
        ],
        "insight": "🚨 ₹1.05 CRORE in untraceable cash — the #1 profit killer. Bank charged TDS u/s 194N (>₹1Cr threshold). Every ₹1 in cash costs ₹1.30 in real terms (lost GST credit + tax risk). Largest single withdrawal: ₹7.6 lakh.",
        "benchmark": "Well-run construction companies keep cash <5% of turnover. You're at 10.4% — double the safe threshold.",
    },
    {
        "name": "GST & Taxes",
        "amount": 7_333_715,
        "pct": 8.4,
        "color": "#10b981",
        "icon": "🏛️",
        "vendors": [
            ("GST Payments (RBIS0GSTPMT)", 7_203_285),
            ("Income Tax / CBDT", 128_850),
            ("TDS on Cash Withdrawal (194N)", 1_580),
        ],
        "insight": "GST of ₹72L on ₹10Cr turnover = 7.2% effective. But payments are irregular — ₹23.3L single payment on 13-Aug suggests 3-4 months accumulated, likely attracting 18% interest + penalties of ₹2-3L/year.",
        "benchmark": "Monthly GST filing would save ₹2-3L/year in interest and penalties.",
    },
    {
        "name": "Related Party / Suspicious",
        "amount": 5_257_900,
        "pct": 6.0,
        "color": "#dc2626",
        "icon": "🔴",
        "vendors": [
            ("Rajender Kumar (2 txns) ⚠️", 2_200_000),
            ("Ashish Sharma (net outflow) ⚠️", 1_837_900),
            ("Kamal Sharma (net outflow) ⚠️", 1_220_000),
        ],
        "insight": "🚨 ₹52.6L to 3 individuals with NO clear business purpose. Rajender Kumar's ₹17L single transfer is most suspicious. Ashish Sharma appears to be a promoter using company account for personal expenses.",
        "benchmark": "Related party transactions should be ZERO unless properly documented and at arm's length.",
    },
    {
        "name": "EMI & Loan Servicing",
        "amount": 2_263_907,
        "pct": 2.6,
        "color": "#6366f1",
        "icon": "🏦",
        "vendors": [
            ("EMI 138559543 (₹1.21L/month)", 1_455_948),
            ("EMI 139866256 (₹34K/month)", 408_564),
            ("O/S Interest Recovery (overdue)", 249_395),
            ("IMPS Loan (Ivy Bank)", 150_000),
        ],
        "insight": "₹1.55L/month fixed EMI regardless of revenue. 'O/S Interest Recovery' entries mean OVERDUE loan payments attracting additional interest. This is a debt-spiral warning sign.",
        "benchmark": "Debt servicing at 2.3% of turnover is manageable, but the overdue interest entries are concerning.",
    },
    {
        "name": "Inter-Account Transfers",
        "amount": 2_280_000,
        "pct": 2.6,
        "color": "#a855f7",
        "icon": "🔄",
        "vendors": [
            ("Kwality Construction (50200079184983)", 2_280_000),
        ],
        "insight": "₹29.5L transferred to own account on 26-Mar-25 (year-end). Classic window dressing pattern. Net ₹22.8L outflow suggests the other account may have liabilities being serviced.",
        "benchmark": "Self-transfers should net to zero. Persistent net outflow suggests hidden liabilities.",
    },
    {
        "name": "Overheads & Office",
        "amount": 1_983_225,
        "pct": 2.3,
        "color": "#14b8a6",
        "icon": "🏢",
        "vendors": [
            ("Kwality Mart P Ltd (related?)", 575_000),
            ("Jaigun Furniture", 200_000),
            ("Dehradun Premier Motors", 190_857),
            ("Fern Brentwood Hotel", 94_558),
            ("Balaji Lites", 63_123),
            ("Other Overheads", 859_687),
        ],
        "insight": "Kwality Mart P Ltd (₹5.75L) may be a related entity. Hotel stay of ₹94K needs justification. ₹1.9L to Dehradun Premier Motors should be capitalized, not expensed.",
        "benchmark": "Overheads at 2% is lean. Watch for personal expenses hiding here.",
    },
    {
        "name": "Equipment & Vehicles",
        "amount": 989_993,
        "pct": 1.1,
        "color": "#0ea5e9",
        "icon": "🚛",
        "vendors": [
            ("Hanumant Homes Pvt Ltd", 185_968),
            ("Soshil Motors", 171_950),
            ("Dashmesh Automobiles", 105_400),
            ("Mine Stone Developers", 98_280),
            ("Ambassador Bldg Solutions", 84_376),
            ("Other Equipment", 344_019),
        ],
        "insight": "Very low equipment spend suggests heavy reliance on rented equipment (hidden in sub-contractor costs) or labor-intensive operations with minimal mechanization.",
        "benchmark": "Equipment at 1% is very low for a ₹10Cr company. Likely hidden in sub-contractor costs.",
    },
]

# ════════════════════════════════════════════════════════════════
# REVENUE SOURCES
# ════════════════════════════════════════════════════════════════
REVENUE_SOURCES = [
    ("Wynberg Allen School", 32_409_500, 32.5, "#2563eb"),
    ("Woodstock School", 25_837_022, 25.9, "#3b82f6"),
    ("Radhika Prakash (Promoter?)", 15_286_813, 15.3, "#60a5fa"),
    ("Govt — Garhwal Central Div", 11_753_250, 11.8, "#059669"),
    ("Govt — Shimla Central Div", 7_155_197, 7.2, "#10b981"),
    ("Medielect Group", 4_724_000, 4.7, "#8b5cf6"),
    ("Govt — Mussoorie Project Div", 1_137_970, 1.1, "#34d399"),
    ("Other Sources", 1_450_878, 1.5, "#94a3b8"),
]

# ════════════════════════════════════════════════════════════════
# SUSPICIOUS TRANSACTIONS
# ════════════════════════════════════════════════════════════════
SUSPICIOUS = [
    {
        "severity": "CRITICAL",
        "entity": "Noorkhan S/O Gulam Rasool",
        "account": "50100375355702",
        "amount": 1_839_350,
        "count": 111,
        "avg": 16_571,
        "detail": "Nearly DAILY payments (111 in 365 days). Average ₹16,571 per transaction. Personal savings account, not business. Payments continued even when account balance was near ₹0. No GST invoices, no work orders.",
        "risk": "Ghost Employee / Fund Siphoning",
    },
    {
        "severity": "CRITICAL",
        "entity": "Cash Withdrawals (SELF/AMIT/SUMIT)",
        "account": "Cash",
        "amount": 10_555_000,
        "count": 101,
        "avg": 104_505,
        "detail": "₹1.05Cr untraceable cash. TDS u/s 194N charged. Largest: ₹7.6L. Cash payments can't be claimed as expenses, lose GST credit, attract tax risk.",
        "risk": "Tax Evasion / Unaccounted Payments",
    },
    {
        "severity": "HIGH",
        "entity": "Rajender Kumar",
        "account": "99909805038345",
        "amount": 2_200_000,
        "count": 2,
        "avg": 1_100_000,
        "detail": "₹5L (21-Sep-24) + ₹17L (17-Jan-25) to personal savings account. No matching business narration. 999-prefix account = personal savings.",
        "risk": "Unauthorized Fund Diversion",
    },
    {
        "severity": "HIGH",
        "entity": "Ashish Sharma",
        "account": "27717610000101 + 27711000002306",
        "amount": 2_577_900,
        "count": 36,
        "avg": 71_608,
        "detail": "Net outflow ₹18.4L (Out: ₹25.8L, In: ₹7.4L). Multiple accounts, bidirectional flows. Large transfers of ₹3.5L, ₹2.6L, ₹1.9L.",
        "risk": "Personal Expenses via Company",
    },
    {
        "severity": "HIGH",
        "entity": "Account Balance — Hit Zero",
        "account": "N/A",
        "amount": 0,
        "count": 12,
        "avg": 0,
        "detail": "Balance = ₹0 on 04-Sep-24 and 04-Mar-25. Below ₹10K on 8+ occasions. Below ₹50K on 15+ occasions. ₹10Cr turnover company with zero cash reserves.",
        "risk": "Operational Insolvency Risk",
    },
    {
        "severity": "HIGH",
        "entity": "Umesh (Labor Contractor?)",
        "account": "SBIN0016121",
        "amount": 2_281_263,
        "count": 46,
        "avg": 49_593,
        "detail": "₹22.8L across 46 transactions (almost weekly). Range ₹5K to ₹2.67L. No proper billing pattern visible.",
        "risk": "Undocumented Labor Payments",
    },
    {
        "severity": "MEDIUM",
        "entity": "Kamal Sharma",
        "account": "27711930000075",
        "amount": 2_020_000,
        "count": 12,
        "avg": 168_333,
        "detail": "All round amounts. Out: ₹20.2L, In: ₹8L. Net: ₹12.2L. Bidirectional flows suggest accommodation entries.",
        "risk": "Accommodation Entries / Fund Routing",
    },
    {
        "severity": "MEDIUM",
        "entity": "GST Payments (Irregular)",
        "account": "RBIS0GSTPMT",
        "amount": 7_203_285,
        "count": 11,
        "avg": 654_844,
        "detail": "₹72L paid but irregularly. ₹23.3L single payment on 13-Aug suggests 3-4 months accumulated. Late filing = 18% interest + penalties.",
        "risk": "GST Penalties / Compliance Risk",
    },
]

# ════════════════════════════════════════════════════════════════
# RECOMMENDATIONS (for profit roadmap)
# ════════════════════════════════════════════════════════════════
RECOMMENDATIONS = [
    {
        "section": "🚨 Stop the Bleeding — Immediate",
        "color": "#dc2626",
        "potential": "₹28-35L/year",
        "items": [
            ("Investigate & Stop Noorkhan Payments", "₹18.4L/year", "This Week", "Demand invoices for all 111 transactions. If undocumented, stop immediately. If legit labor, convert to proper contract with GST."),
            ("Eliminate Cash Withdrawals (>₹50K)", "₹13-15L/year", "2 Weeks", "Move ALL payments to UPI/NEFT. For labor wages, use PayNearby/PayTM Business. Cash costs you 30% more than digital."),
            ("Investigate Rajender Kumar ₹22L", "₹22L recovery", "This Week", "Obtain documents within 7 days. If no valid reason, issue legal notice for recovery."),
            ("Separate Personal Expenses", "₹30.6L recovered", "2 Weeks", "Create Partner Drawing account. Ashish Sharma (₹18.4L) + Kamal Sharma (₹12.2L) = ₹30.6L questionable outflows."),
        ],
    },
    {
        "section": "💰 Reduce Costs — Operational",
        "color": "#ea580c",
        "potential": "₹40-60L/year",
        "items": [
            ("Consolidate Material Vendors", "₹20-30L/year", "1 Month", "You pay 30+ vendors. Consolidate to 3-4 with annual rate contracts. ₹2.69Cr materials × 8-10% = ₹20-27L savings."),
            ("Reduce Sub-contracting 30%", "₹15-20L/year", "3 Months", "Sub-contractors (₹1.83Cr = 21%) = huge margin loss. Build in-house capability for concrete, plastering, finishing."),
            ("Fix GST — File Monthly", "₹2-3L/year", "1 Month", "₹23.3L lump payment = months accumulated. At 18% late fee + penalties → ₹2-3L wasted annually."),
            ("Claim Full GST Input Credit", "₹10-15L/year", "2 Months", "Get GST invoices from all vendors. On ₹80L+ eligible spend, 18% ITC = ₹10-15L recovered."),
            ("Renegotiate Fuel Costs", "₹2-3L/year", "2 Weeks", "₹22.2L on fuel. Fleet card with 2-3% discount + GPS tracking to prevent pilferage."),
        ],
    },
    {
        "section": "📈 Grow Revenue — Top Line",
        "color": "#059669",
        "potential": "₹50L-1Cr additional",
        "items": [
            ("Diversify Beyond Schools", "+₹30-50L revenue", "6 Months", "58% revenue from 2 schools. Loss of either = insolvency in 2 months. Bid for PWD tenders in Dehradun/Haridwar."),
            ("Speed Up Govt Bill Collection", "+₹5-10L cash flow", "1 Month", "₹2Cr+ from govt comes in irregular lumps. Dedicate person for follow-up. File interim bills monthly."),
            ("Add Maintenance Contracts (AMC)", "+₹15-20L/year", "3 Months", "Offer AMC at 2-3% of construction value to existing school + govt clients. Creates stable monthly income."),
            ("Price at 15-20% Margin Minimum", "+₹50L-1Cr profit", "Ongoing", "₹10Cr turnover with net LOSS means pricing is wrong. Every bid must include 15% minimum profit margin."),
        ],
    },
    {
        "section": "🏦 Fix Cash Flow — Stability",
        "color": "#2563eb",
        "potential": "Prevents insolvency",
        "items": [
            ("Maintain ₹5L Minimum Balance", "Risk mitigation", "1 Week", "Balance hit ₹0 twice. Set standing instruction: freeze non-essential payments below ₹5L. Get OD limit of ₹10-15L."),
            ("Weekly Payment Cycle (not daily)", "Better control", "2 Weeks", "You make 1,130 debits (4/day). Batch to weekly: Fridays for vendors, Mondays for labor. Better forecasting."),
            ("Refinance the ₹1.21L EMI", "₹3-5L/year", "1 Month", "If current rate is 12-14%, refinance at 9-10% (SBI/BOB). O/S interest entries suggest penalty territory already."),
            ("Bill Discounting Facility", "Cash flow smoothing", "1 Month", "₹4.5Cr+ from schools (creditworthy). Get 80-90% invoice value immediately. Cost: 8-10%/yr, prevents zero-balance crises."),
        ],
    },
]

# ════════════════════════════════════════════════════════════════
# PROFIT BRIDGE
# ════════════════════════════════════════════════════════════════
PROFIT_BRIDGE = {
    "current_loss": -13_61_726,
    "stop_leakages": 28_00_000,
    "cost_reduction": 50_00_000,
    "revenue_growth": 30_00_000,
    "projected_profit": 94_38_274,
}

SYMBOLS = [
        # Mega-cap Technology
    "NVDA",     # NVIDIA Corporation
    "AAPL",     # Apple Inc.
    "MSFT",     # Microsoft Corporation
    "GOOGL",    # Alphabet Inc. (Class A)
    "AMZN",     # Amazon.com, Inc.
    "META",     # Meta Platforms, Inc.
    "TSM",      # Taiwan Semiconductor Manufacturing Co.
    "ASML",     # ASML Holding
    "ORCL",     # Oracle Corporation
    "IBM",      # IBM Corporation
        # Financials
    "BRK-B",    # Berkshire Hathaway Inc.
    "JPM",      # JPMorgan Chase & Co.
    "V",        # Visa Inc.
    "MA",       # Mastercard Inc.
    "BAC",      # Bank of America Corp.
    "GS",       # Goldman Sachs Group
    "MS",       # Morgan Stanley
    "WFC",      # Wells Fargo & Co.
    "C",        # Citigroup Inc.

    # Consumer Staples / Retail / Discretionary
    "WMT",      # Walmart Inc.
    "PG",       # Procter & Gamble Co.
    "KO",       # Coca-Cola Co.
    "PEP",      # PepsiCo, Inc.
    "COST",     # Costco Wholesale Corp.
    "HD",       # Home Depot, Inc.
    "MCD",      # McDonald's Corp.
    "SBUX",     # Starbucks Corp.
    "NKE",      # Nike, Inc.

    # Healthcare
    "UNH",      # UnitedHealth Group Inc.
    "JNJ",      # Johnson & Johnson
    "PFE",      # Pfizer Inc.
    "MRK",      # Merck & Co., Inc.
    "LLY",      # Eli Lilly & Co.
    "ABBV",     # AbbVie Inc.

    # Energy & Commodities
    "XOM",      # Exxon Mobil Corp.
    "CVX",      # Chevron Corp.
    "SHEL",     # Shell plc
    "BP",       # BP plc
    "TTE",      # TotalEnergies SE
    "COP",      # ConocoPhillips
    "SLB",      # Schlumberger Ltd.

    # Industrials & Infrastructure
    "BA",       # Boeing Co.
    "CAT",      # Caterpillar Inc.
    "GE",       # General Electric Co.
    "HON",      # Honeywell International Inc.
    "UPS",      # United Parcel Service, Inc.
    "FDX",      # FedEx Corporation
    "MMM",      # 3M Co.
    "RTN",      # Raytheon Technologies Corp.

    # Global Tech / Consumer Leaders (Intl.)
    "BABA",     # Alibaba Group Holding
    "TCEHY",    # Tencent Holdings
    "SAP",      # SAP SE
    "TM",       # Toyota Motor Corp.
    "005930.KS",# Samsung Electronics (KRX)
    "NESN.SW",  # Nestlé SA (SWX)
    "ROG.SW",   # Roche Holding AG
    "MC.PA",    # LVMH Moët Hennessy

    # Telecom / Media
    "T",        # AT&T Inc.
    "VZ",       # Verizon Communications Inc.
    "TMUS",     # T-Mobile US, Inc.
    "CMCSA",    # Comcast Corp.
    "NFLX",     # Netflix, Inc.

    # Crypto
    "BTC/USD",  # Bitcoin
    "ETH/USD",  # Ethereum

    # Materials / Mining
    "BHP",      # BHP Group
    "RIO",      # Rio Tinto plc
    "MT",       # ArcelorMittal S.A.
    "DD",       # DuPont de Nemours, Inc.
    "VALE",     # Vale S.A.

    # International Financials
    "HSBC",     # HSBC Holdings plc
    "UBS",      # UBS Group AG
    "ICBC",     # Industrial and Commercial Bank of China

    # Smaller Emerging / Fintech / Misc
    "WELL",     # WELL Health Technologies
    "BNBBUSD"   # Binance Coin (as a USD pair)
]

FRED_INDICATORS = [
    # 1️⃣ High Predictive Power (Equities / Credit Signals)
    "IGREA",                # Global Real Economic Activity Index
    "WUIGLOBALWEIGHTAVG",   # World Uncertainty Index (GDP Weighted)
    "WUIGLOBALSMPAVG",      # World Uncertainty Index (Simple Avg)
    "PALLFNFINDEXQ",        # Global Price Index of All Commodities (broad macro stress proxy)

    # 2️⃣ Macro Trend Indicators
    "NYGDPMKTPCDWLD",       # World GDP (current USD)
    "NYGDPPCAPKDWLD",       # World GDP per capita (real)
    "FPCPITOTLZGWLD",       # Global CPI (inflation)
    "SLUEM1524ZSWLD",       # World Youth Unemployment

    # 3️⃣ Commodity & Market Sensitivity (Priority 3)
    "PNRGINDEXM",           # Global Energy Price Index
    "PFOODINDEXM",          # Global Food Price Index
    "PINDUINDEXM",          # Global Industrial Materials Index
    "PMETAINDEXM",          # Global Metals Index
    "PCOPPUSDM",            # Copper price
    "POILBREUSDM",          # Brent Crude
    "POILWTIUSDM",          # WTI Crude
    "POILDUBUSDM"           # Dubai Crude
]
TOBE=[
        # Mega-cap Technology
    "NVDA",     # NVIDIA Corporation
    "AAPL",     # Apple Inc.
    "MSFT",     # Microsoft Corporation
    "GOOGL",    # Alphabet Inc. (Class A)
    "AMZN",     # Amazon.com, Inc.
    "META",     # Meta Platforms, Inc.
    "TSM",      # Taiwan Semiconductor Manufacturing Co.
    "ASML",     # ASML Holding
    "ORCL",     # Oracle Corporation
    "IBM",      # IBM Corporation
        # Financials
    "BRK-B",    # Berkshire Hathaway Inc.
    "JPM",      # JPMorgan Chase & Co.
    "V",        # Visa Inc.
    "MA",       # Mastercard Inc.
    "BAC",      # Bank of America Corp.
    "GS",       # Goldman Sachs Group
    "MS",       # Morgan Stanley
    "WFC",      # Wells Fargo & Co.
    "C",        # Citigroup Inc.

    # Consumer Staples / Retail / Discretionary
    "WMT",      # Walmart Inc.
    "PG",       # Procter & Gamble Co.
    "KO",       # Coca-Cola Co.
    "PEP",      # PepsiCo, Inc.
    "COST",     # Costco Wholesale Corp.
    "HD",       # Home Depot, Inc.
    "MCD",      # McDonald's Corp.
    "SBUX",     # Starbucks Corp.
    "NKE",      # Nike, Inc.

    # Healthcare
    "UNH",      # UnitedHealth Group Inc.
    "JNJ",      # Johnson & Johnson
    "PFE",      # Pfizer Inc.
    "MRK",      # Merck & Co., Inc.
    "LLY",      # Eli Lilly & Co.
    "ABBV",     # AbbVie Inc.

    # Energy & Commodities
    "XOM",      # Exxon Mobil Corp.
    "CVX",      # Chevron Corp.
    "SHEL",     # Shell plc
    "BP",       # BP plc
    "TTE",      # TotalEnergies SE
    "COP",      # ConocoPhillips
    "SLB",      # Schlumberger Ltd.

    # Industrials & Infrastructure
    "BA",       # Boeing Co.
    "CAT",      # Caterpillar Inc.
    "GE",       # General Electric Co.
    "HON",      # Honeywell International Inc.
    "UPS",      # United Parcel Service, Inc.
    "FDX",      # FedEx Corporation
    "MMM",      # 3M Co.
    "RTN",      # Raytheon Technologies Corp.

    # Global Tech / Consumer Leaders (Intl.)
    "BABA",     # Alibaba Group Holding
    "TCEHY",    # Tencent Holdings
    "SAP",      # SAP SE
    "TM",       # Toyota Motor Corp.
    "005930.KS",# Samsung Electronics (KRX)
    "NESN.SW",  # Nestlé SA (SWX)
    "ROG.SW",   # Roche Holding AG
    "MC.PA",    # LVMH Moët Hennessy

    # Telecom / Media
    "T",        # AT&T Inc.
    "VZ",       # Verizon Communications Inc.
    "TMUS",     # T-Mobile US, Inc.
    "CMCSA",    # Comcast Corp.
    "NFLX",     # Netflix, Inc.

    # Crypto
    "BTC/USD",  # Bitcoin
    "ETH/USD",  # Ethereum

    # Materials / Mining
    "BHP",      # BHP Group
    "RIO",      # Rio Tinto plc
    "MT",       # ArcelorMittal S.A.
    "DD",       # DuPont de Nemours, Inc.
    "VALE",     # Vale S.A.

    # International Financials
    "HSBC",     # HSBC Holdings plc
    "UBS",      # UBS Group AG
    "ICBC",     # Industrial and Commercial Bank of China

    # Smaller Emerging / Fintech / Misc
    "WELL",     # WELL Health Technologies
    "BNBBUSD"   # Binance Coin (as a USD pair)
]

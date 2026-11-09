QUEUES = ["billing", "delivery", "technical", "security", "general", "spam"]
PRIORITIES = ["high", "normal"]

QUEUE_DEFINITIONS = {
    "billing": "Payments and money: charges, double charges, declined payments, refunds of money, "
               "promo codes or discounts not applied to the price, invoices, installments, "
               "subscription fees, bonus points not credited.",
    "delivery": "Shipping: tracking, late, lost, missing, damaged or wrong parcels, courier behaviour, "
                "changing the delivery address or time, pickup-point and courier delivery options, "
                "parcels returned to the sender.",
    "technical": "Bugs in the app, website or seller dashboard: crashes, errors, pages or buttons that "
                 "do not work, SMS codes not arriving, display or translation problems.",
    "security": "Someone other than the customer accessed or changed the account, unauthorized orders "
                "or payments, stolen bonus points, phishing or scam attempts against the customer "
                "(fake support calls, asking for codes or card data, paying a seller off-platform), "
                "lost device that is still logged in.",
    "general": "Questions about products, policies, gift cards, opening hours, selling on the platform, "
               "how-to questions, recommendations, feedback and thanks.",
    "spam": "Not a customer asking for help: advertising, unsolicited business offers, loans, job ads, "
            "prize scams, selling unrelated things, test or random text.",
}

SECURITY_T = 0.5        
URGENCY_HIGH_T = 1.5    
INJECTION_T = 0.5       
TARGET_ACCURACY = 0.95  

PRICES = {
    "jev": {"input": 0.042, "output": 0.0},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},  
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast": {"input": 0.293, "output": 2.253},
}

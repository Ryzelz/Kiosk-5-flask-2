"""PayMongo QR Ph helper - wraps the v1 Payment Intent + Payment Method flow."""

import os
import requests
from base64 import b64encode

PAYMONGO_SECRET_KEY = os.environ.get('PAYMONGO_SECRET_KEY', '')
PAYMONGO_RETURN_URL = os.environ.get('PAYMONGO_RETURN_URL', '').strip()

API_BASE = 'https://api.paymongo.com/v1'


def _headers():
    if not PAYMONGO_SECRET_KEY:
        raise RuntimeError('PAYMONGO_SECRET_KEY is not configured.')

    encoded = b64encode(f'{PAYMONGO_SECRET_KEY}:'.encode()).decode()
    return {
        'Authorization': f'Basic {encoded}',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
    }


def create_payment_intent(amount_php: float, description: str = 'Purchase of goods'):
    """Create a PaymentIntent. *amount_php* is in pesos (e.g. 150.00)."""
    amount_centavos = int(round(amount_php * 100))

    payload = {
        'data': {
            'attributes': {
                'amount': amount_centavos,
                'payment_method_allowed': ['qrph'],
                'currency': 'PHP',
                'description': description,
                'statement_descriptor': 'Wideye Kiosk',
            }
        }
    }

    resp = requests.post(f'{API_BASE}/payment_intents', json=payload, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()['data']


def attach_qrph(payment_intent_id: str):
    """Create a QR Ph payment method and attach it to *payment_intent_id*.

    Returns the full PaymentIntent resource whose ``next_action`` field
    contains the QR Ph image URL the customer must scan.
    """
    pm_payload = {
        'data': {
            'attributes': {
                'type': 'qrph',
            }
        }
    }

    pm_resp = requests.post(f'{API_BASE}/payment_methods', json=pm_payload, headers=_headers(), timeout=30)
    pm_resp.raise_for_status()
    payment_method_id = pm_resp.json()['data']['id']

    attach_attributes = {
        'payment_method': payment_method_id,
    }

    if PAYMONGO_RETURN_URL:
        attach_attributes['return_url'] = PAYMONGO_RETURN_URL

    attach_payload = {
        'data': {
            'attributes': attach_attributes
        }
    }

    attach_resp = requests.post(
        f'{API_BASE}/payment_intents/{payment_intent_id}/attach',
        json=attach_payload,
        headers=_headers(),
        timeout=30,
    )
    attach_resp.raise_for_status()
    return attach_resp.json()['data']


def retrieve_payment_intent(payment_intent_id: str):
    """Poll a PaymentIntent to check its current status."""
    resp = requests.get(f'{API_BASE}/payment_intents/{payment_intent_id}', headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()['data']

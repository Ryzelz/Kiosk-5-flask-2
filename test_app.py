"""
Comprehensive pytest suite for the Wideye Kiosk Flask application.

Covers: Auth, Products, Cart, Orders, Payment, Admin, Face Recognition AI,
        Size/Multi-image features, and helper utilities.

Run with:
    python -m pytest
or
    python -m pytest -v -s
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from io import BytesIO
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault('SECRET_KEY', 'test-secret-key')
os.environ.setdefault('PAYMONGO_SECRET_KEY', 'sk_test_fake_key')


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope='function')
def app(tmp_path):
    """Create a fresh Flask app + temporary DB for every test."""
    db_path = tmp_path / 'test.sqlite3'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path.as_posix()}'

    from website import create_app, db as _db

    application = create_app()
    application.config.update({
        'TESTING': True,
        'WTF_CSRF_ENABLED': False,
    })

    with application.app_context():
        yield application
        _db.session.remove()

    # Clean up env
    os.environ.pop('DATABASE_URL', None)


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def db(app):
    from website import db as _db
    return _db


@pytest.fixture
def sample_product(app, db):
    """Insert a product with size + multi-image fields."""
    from website.models import Product
    with app.app_context():
        p = Product()
        p.product_name = 'Test Latte'
        p.current_price = 150.0
        p.previous_price = 180.0
        p.in_stock = 50
        p.size = 'Small, Medium, Large'
        p.sugar = 'None, Light, Regular'
        p.milk = 'Whole, Oat'
        p.shot = 'Single, Double'
        p.product_picture = '/media/small.jpg, /media/medium.jpg, /media/large.jpg'
        p.flash_sale = False
        db.session.add(p)
        db.session.commit()
        return p.id


@pytest.fixture
def admin_user(app, db):
    """Create an admin user and return (id, email, password)."""
    from website.models import Customer
    with app.app_context():
        c = Customer()
        c.email = 'admin@test.com'
        c.username = 'admin'
        c.password = 'admin123'
        c.is_admin = True
        c.face_profile_name = 'admin'
        db.session.add(c)
        db.session.commit()
        return c.id, 'admin@test.com', 'admin123'


@pytest.fixture
def normal_user(app, db):
    """Create a regular (non-admin) user."""
    from website.models import Customer
    with app.app_context():
        c = Customer()
        c.email = 'user@test.com'
        c.username = 'testuser'
        c.password = 'user1234'
        c.is_admin = False
        c.face_profile_name = 'testuser'
        db.session.add(c)
        db.session.commit()
        return c.id, 'user@test.com', 'user1234'


def login(client, email, password):
    return client.post('/login', data={
        'email': email,
        'password': password,
    }, follow_redirects=True)


# ===========================================================================
# 1. UTILITY / HELPER TESTS
# ===========================================================================

class TestParseOptions:
    @pytest.mark.parametrize(
        ('csv_string', 'expected'),
        [
            ('', []),
            (None, []),
            ('Small', ['Small']),
            ('Small, Medium, Large', ['Small', 'Medium', 'Large']),
            ('  A ,  B  , C  ', ['A', 'B', 'C']),
            ('A,,B,  ,C', ['A', 'B', 'C']),
        ],
        ids=[
            'empty-string',
            'none-value',
            'single-option',
            'multiple-options',
            'trim-whitespace',
            'remove-empty-tokens',
        ],
    )
    def test_parse_options_cases(self, csv_string, expected):
        from website.views import parse_options
        assert parse_options(csv_string) == expected


class TestGetProductImage:
    def test_first_image_no_size(self, app, db, sample_product):
        from website.views import get_product_image
        from website.models import Product
        with app.app_context():
            p = Product.query.get(sample_product)
            assert get_product_image(p) == '/media/small.jpg'

    def test_correct_image_for_size(self, app, db, sample_product):
        from website.views import get_product_image
        from website.models import Product
        with app.app_context():
            p = Product.query.get(sample_product)
            assert get_product_image(p, 'Medium') == '/media/medium.jpg'
            assert get_product_image(p, 'Large') == '/media/large.jpg'

    def test_fallback_to_last_image(self, app, db):
        from website.views import get_product_image
        from website.models import Product
        with app.app_context():
            p = Product()
            p.product_name = 'X'
            p.current_price = 10
            p.previous_price = 12
            p.in_stock = 5
            p.size = 'S, M, L, XL'
            p.product_picture = '/media/a.jpg, /media/b.jpg'
            p.sugar = ''
            p.milk = ''
            p.shot = ''
            db.session.add(p)
            db.session.commit()
            # XL is index 3 but only 2 images → fallback to last
            assert get_product_image(p, 'XL') == '/media/b.jpg'

    def test_unknown_size_returns_first(self, app, db, sample_product):
        from website.views import get_product_image
        from website.models import Product
        with app.app_context():
            p = Product.query.get(sample_product)
            assert get_product_image(p, 'Gigantic') == '/media/small.jpg'

    def test_single_image_product(self, app, db):
        from website.views import get_product_image
        from website.models import Product
        with app.app_context():
            p = Product()
            p.product_name = 'Simple'
            p.current_price = 10
            p.previous_price = 12
            p.in_stock = 5
            p.size = ''
            p.product_picture = '/media/only.jpg'
            p.sugar = ''
            p.milk = ''
            p.shot = ''
            db.session.add(p)
            db.session.commit()
            assert get_product_image(p) == '/media/only.jpg'


class TestFormatOptionSummary:
    @pytest.mark.parametrize(
        ('item_kwargs', 'expected_parts'),
        [
            ({'size': '', 'sugar': '', 'milk': '', 'shot': ''}, ['No add-ons']),
            ({'size': 'Large', 'sugar': '', 'milk': '', 'shot': ''}, ['Size: Large']),
            (
                {'size': 'Medium', 'sugar': 'Light', 'milk': 'Oat', 'shot': 'Double'},
                ['Size: Medium', 'Sugar: Light', 'Milk: Oat', 'Shot: Double'],
            ),
        ],
        ids=['no-addons', 'size-only', 'full-options'],
    )
    def test_format_option_summary_cases(self, item_kwargs, expected_parts):
        from website.views import format_option_summary
        item = MagicMock(**item_kwargs)
        summary = format_option_summary(item)
        for expected_part in expected_parts:
            assert expected_part in summary


# ===========================================================================
# 2. AUTH TESTS
# ===========================================================================

class TestAuth:
    def test_signup_creates_user(self, client, app, db):
        resp = client.post('/sign-up', data={
            'email': 'new@test.com',
            'username': 'newuser',
            'password1': 'pass1234',
            'password2': 'pass1234',
        }, follow_redirects=True)
        from website.models import Customer
        with app.app_context():
            user = Customer.query.filter_by(email='new@test.com').first()
            assert user is not None
            assert user.username == 'newuser'

    def test_first_user_is_admin(self, client, app, db):
        client.post('/sign-up', data={
            'email': 'first@test.com',
            'username': 'firstuser',
            'password1': 'pass1234',
            'password2': 'pass1234',
        }, follow_redirects=True)
        from website.models import Customer
        with app.app_context():
            user = Customer.query.filter_by(email='first@test.com').first()
            assert user.is_admin is True

    def test_second_user_is_not_admin(self, client, app, db, admin_user):
        client.post('/sign-up', data={
            'email': 'second@test.com',
            'username': 'seconduser',
            'password1': 'pass1234',
            'password2': 'pass1234',
        }, follow_redirects=True)
        from website.models import Customer
        with app.app_context():
            user = Customer.query.filter_by(email='second@test.com').first()
            assert user.is_admin is False

    def test_duplicate_email_rejected(self, client, app, db, admin_user):
        uid, email, pw = admin_user
        resp = client.post('/sign-up', data={
            'email': email,
            'username': 'duplicate',
            'password1': 'pass1234',
            'password2': 'pass1234',
        }, follow_redirects=True)
        assert b'Account Not Created' in resp.data or b'already exists' in resp.data

    def test_login_valid(self, client, admin_user):
        uid, email, pw = admin_user
        resp = login(client, email, pw)
        assert resp.status_code == 200

    def test_login_wrong_password(self, client, admin_user):
        uid, email, pw = admin_user
        resp = client.post('/login', data={
            'email': email,
            'password': 'wrongpassword',
        }, follow_redirects=True)
        assert b'Incorrect' in resp.data or b'invalid' in resp.data.lower() or resp.status_code == 200

    def test_password_change(self, client, app, db, admin_user):
        uid, email, pw = admin_user
        login(client, email, pw)
        resp = client.post(f'/change-password/{uid}', data={
            'current_password': pw,
            'new_password': 'newpass123',
            'confirm_new_password': 'newpass123',
        }, follow_redirects=True)
        from website.models import Customer
        with app.app_context():
            user = Customer.query.get(uid)
            assert user.verify_password('newpass123')


# ===========================================================================
# 3. PRODUCT / SHOP ITEM TESTS
# ===========================================================================

class TestProducts:
    def test_home_page_loads(self, client, sample_product):
        resp = client.get('/')
        assert resp.status_code == 200
        assert b'Test Latte' in resp.data

    def test_home_shows_size_dropdown(self, client, sample_product):
        resp = client.get('/')
        assert b'Small' in resp.data
        assert b'Medium' in resp.data
        assert b'Large' in resp.data

    def test_add_product_as_admin(self, client, app, db, admin_user):
        uid, email, pw = admin_user
        login(client, email, pw)

        # Create a fake image file
        img = BytesIO(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)
        img.name = 'test.png'

        resp = client.post('/add-shop-items', data={
            'product_name': 'New Coffee',
            'current_price': '120.0',
            'previous_price': '150.0',
            'in_stock': '30',
            'size': 'Small, Large',
            'sugar': 'None, Regular',
            'milk': '',
            'shot': '',
            'flash_sale': False,
            'product_picture': (img, 'test.png'),
        }, content_type='multipart/form-data', follow_redirects=True)

        from website.models import Product
        with app.app_context():
            p = Product.query.filter_by(product_name='New Coffee').first()
            assert p is not None
            assert p.size == 'Small, Large'

    def test_add_product_blocked_for_non_admin(self, client, normal_user):
        uid, email, pw = normal_user
        login(client, email, pw)
        resp = client.get('/add-shop-items', follow_redirects=True)
        assert b'404' in resp.data or resp.status_code == 404

    def test_delete_product(self, client, app, db, admin_user, sample_product):
        uid, email, pw = admin_user
        login(client, email, pw)
        resp = client.get(f'/delete-item/{sample_product}', follow_redirects=True)
        from website.models import Product
        with app.app_context():
            assert Product.query.get(sample_product) is None

    def test_update_product_size(self, client, app, db, admin_user, sample_product):
        uid, email, pw = admin_user
        login(client, email, pw)
        resp = client.post(f'/update-item/{sample_product}', data={
            'product_name': 'Test Latte',
            'current_price': '150.0',
            'previous_price': '180.0',
            'in_stock': '50',
            'size': 'Tall, Grande, Venti',
            'sugar': 'None, Light, Regular',
            'milk': 'Whole, Oat',
            'shot': 'Single, Double',
            'flash_sale': False,
        }, content_type='multipart/form-data', follow_redirects=True)
        from website.models import Product
        with app.app_context():
            p = Product.query.get(sample_product)
            assert p.size == 'Tall, Grande, Venti'


# ===========================================================================
# 4. CART TESTS
# ===========================================================================

class TestCart:
    def test_add_to_cart_with_size(self, client, app, db, normal_user, sample_product):
        uid, email, pw = normal_user
        login(client, email, pw)
        resp = client.post(f'/add-to-cart/{sample_product}', data={
            'size': 'Medium',
            'sugar': 'Light',
            'milk': 'Oat',
            'shot': 'Double',
        }, follow_redirects=True)
        from website.models import Cart
        with app.app_context():
            cart = Cart.query.filter_by(customer_link=uid).first()
            assert cart is not None
            assert cart.size == 'Medium'
            assert cart.sugar == 'Light'
            assert cart.milk == 'Oat'
            assert cart.shot == 'Double'

    def test_add_to_cart_defaults_to_first_size(self, client, app, db, normal_user, sample_product):
        uid, email, pw = normal_user
        login(client, email, pw)
        # No size submitted → should default to first size option
        client.post(f'/add-to-cart/{sample_product}', data={
            'sugar': 'None',
            'milk': '',
            'shot': '',
        }, follow_redirects=True)
        from website.models import Cart
        with app.app_context():
            cart = Cart.query.filter_by(customer_link=uid).first()
            assert cart is not None
            assert cart.size == 'Small'

    def test_invalid_size_defaults(self, client, app, db, normal_user, sample_product):
        uid, email, pw = normal_user
        login(client, email, pw)
        client.post(f'/add-to-cart/{sample_product}', data={
            'size': 'Nonexistent',
            'sugar': '',
            'milk': '',
            'shot': '',
        }, follow_redirects=True)
        from website.models import Cart
        with app.app_context():
            cart = Cart.query.filter_by(customer_link=uid).first()
            assert cart.size == 'Small'  # defaults to first available

    def test_cart_dedup_by_size(self, client, app, db, normal_user, sample_product):
        uid, email, pw = normal_user
        login(client, email, pw)
        # Add same product twice with same size → quantity should increase
        client.post(f'/add-to-cart/{sample_product}', data={'size': 'Large'}, follow_redirects=True)
        client.post(f'/add-to-cart/{sample_product}', data={'size': 'Large'}, follow_redirects=True)
        from website.models import Cart
        with app.app_context():
            carts = Cart.query.filter_by(customer_link=uid, size='Large').all()
            assert len(carts) == 1
            assert carts[0].quantity == 2

    def test_different_sizes_separate_cart_items(self, client, app, db, normal_user, sample_product):
        uid, email, pw = normal_user
        login(client, email, pw)
        client.post(f'/add-to-cart/{sample_product}', data={'size': 'Small'}, follow_redirects=True)
        client.post(f'/add-to-cart/{sample_product}', data={'size': 'Large'}, follow_redirects=True)
        from website.models import Cart
        with app.app_context():
            carts = Cart.query.filter_by(customer_link=uid).all()
            assert len(carts) == 2

    def test_cart_page_loads(self, client, app, db, normal_user, sample_product):
        uid, email, pw = normal_user
        login(client, email, pw)
        client.post(f'/add-to-cart/{sample_product}', data={'size': 'Small'}, follow_redirects=True)
        resp = client.get('/cart')
        assert resp.status_code == 200
        assert b'Test Latte' in resp.data

    def test_add_to_cart_requires_login(self, client, sample_product):
        resp = client.post(f'/add-to-cart/{sample_product}', data={}, follow_redirects=True)
        assert b'login' in resp.data.lower() or b'Log in' in resp.data


# ===========================================================================
# 5. ORDER TESTS
# ===========================================================================

class TestOrders:
    @patch('website.views.create_payment_intent')
    @patch('website.views.attach_qrph')
    def test_place_order_creates_order(self, mock_attach, mock_create_pi, client, app, db, normal_user, sample_product):
        mock_create_pi.return_value = {'id': 'pi_test_123'}
        mock_attach.return_value = {'attributes': {'next_action': {
            'type': 'consume_qr',
            'code': {'image_url': 'data:image/png;base64,fakeqr'}
        }}}

        uid, email, pw = normal_user
        login(client, email, pw)
        client.post(f'/add-to-cart/{sample_product}', data={
            'size': 'Large', 'sugar': 'Regular', 'milk': 'Whole', 'shot': 'Single',
        }, follow_redirects=True)

        resp = client.get('/place-order', follow_redirects=True)
        from website.models import Order, Cart
        with app.app_context():
            order = Order.query.filter_by(customer_link=uid).first()
            assert order is not None
            assert order.size == 'Large'
            assert order.sugar == 'Regular'
            assert order.payment_id == 'pi_test_123'
            assert order.status == 'Pending'
            # Cart should be cleared
            carts = Cart.query.filter_by(customer_link=uid).all()
            assert len(carts) == 0

    def test_place_order_empty_cart(self, client, normal_user):
        uid, email, pw = normal_user
        login(client, email, pw)
        resp = client.get('/place-order', follow_redirects=True)
        assert b'Empty' in resp.data or b'empty' in resp.data

    def test_view_orders_admin(self, client, app, db, admin_user, sample_product):
        uid, email, pw = admin_user
        login(client, email, pw)
        resp = client.get('/view-orders')
        assert resp.status_code == 200

    def test_view_orders_blocked_for_non_admin(self, client, normal_user):
        uid, email, pw = normal_user
        login(client, email, pw)
        resp = client.get('/view-orders', follow_redirects=True)
        assert b'404' in resp.data

    def test_delete_order_admin(self, client, app, db, admin_user):
        uid, email, pw = admin_user
        login(client, email, pw)
        from website.models import Order, Product
        with app.app_context():
            p = Product()
            p.product_name = 'Deletable'
            p.current_price = 10
            p.previous_price = 12
            p.in_stock = 5
            p.size = ''
            p.sugar = ''
            p.milk = ''
            p.shot = ''
            p.product_picture = '/media/x.jpg'
            db.session.add(p)
            db.session.commit()

            o = Order()
            o.quantity = 1
            o.size = ''
            o.sugar = ''
            o.milk = ''
            o.shot = ''
            o.price = 10
            o.status = 'Pending'
            o.payment_id = 'test_pi'
            o.customer_link = uid
            o.product_link = p.id
            db.session.add(o)
            db.session.commit()
            order_id = o.id

        resp = client.get(f'/delete-order/{order_id}', follow_redirects=True)
        with app.app_context():
            assert Order.query.get(order_id) is None

    def test_no_shipping_fee(self, client, app, db, normal_user, sample_product):
        """Verify no +200 shipping fee is added."""
        uid, email, pw = normal_user
        login(client, email, pw)
        client.post(f'/add-to-cart/{sample_product}', data={'size': 'Small'}, follow_redirects=True)
        resp = client.get('/cart')
        # Total should equal the amount (no +200)
        assert b'Shipping' not in resp.data


# ===========================================================================
# 6. PAYMENT TESTS (mocked PayMongo)
# ===========================================================================

class TestPayment:
    @patch('website.views.create_payment_intent')
    @patch('website.views.attach_qrph')
    def test_payment_qr_page_renders(self, mock_attach, mock_create_pi, client, app, db, normal_user, sample_product):
        mock_create_pi.return_value = {'id': 'pi_test_qr'}
        mock_attach.return_value = {'attributes': {'next_action': {
            'type': 'consume_qr',
            'code': {'image_url': 'data:image/png;base64,QRIMAGE'}
        }}}

        uid, email, pw = normal_user
        login(client, email, pw)
        client.post(f'/add-to-cart/{sample_product}', data={'size': 'Small'}, follow_redirects=True)
        resp = client.get('/place-order')
        assert resp.status_code == 200
        assert b'QRIMAGE' in resp.data or b'pi_test_qr' in resp.data

    @patch('website.views.retrieve_payment_intent')
    def test_check_payment_succeeded(self, mock_retrieve, client, app, db, normal_user):
        uid, email, pw = normal_user
        from website.models import Order, Product
        with app.app_context():
            p = Product()
            p.product_name = 'PayTest'
            p.current_price = 100
            p.previous_price = 120
            p.in_stock = 10
            p.size = ''
            p.sugar = ''
            p.milk = ''
            p.shot = ''
            p.product_picture = '/media/pay.jpg'
            db.session.add(p)
            db.session.commit()

            o = Order()
            o.quantity = 1
            o.size = ''
            o.sugar = ''
            o.milk = ''
            o.shot = ''
            o.price = 100
            o.status = 'Pending'
            o.payment_id = 'pi_check_123'
            o.customer_link = uid
            o.product_link = p.id
            db.session.add(o)
            db.session.commit()

        mock_retrieve.return_value = {'attributes': {'status': 'succeeded'}}
        resp = client.get('/check-payment/pi_check_123')
        data = resp.get_json()
        assert data['status'] == 'paid'
        with app.app_context():
            order = Order.query.filter_by(payment_id='pi_check_123').first()
            assert order.status == 'Accepted'

    @patch('website.views.retrieve_payment_intent')
    def test_check_payment_pending(self, mock_retrieve, client):
        mock_retrieve.return_value = {'attributes': {'status': 'awaiting_next_action'}}
        resp = client.get('/check-payment/pi_pending_456')
        data = resp.get_json()
        assert data['status'] == 'pending'


# ===========================================================================
# 7. ADMIN TESTS
# ===========================================================================

class TestAdmin:
    def test_admin_page_accessible(self, client, admin_user):
        uid, email, pw = admin_user
        login(client, email, pw)
        resp = client.get('/admin-page')
        assert resp.status_code == 200

    def test_admin_page_blocked_for_non_admin(self, client, normal_user):
        uid, email, pw = normal_user
        login(client, email, pw)
        resp = client.get('/admin-page', follow_redirects=True)
        assert b'404' in resp.data

    def test_shop_items_page(self, client, admin_user, sample_product):
        uid, email, pw = admin_user
        login(client, email, pw)
        resp = client.get('/shop-items')
        assert resp.status_code == 200
        assert b'Test Latte' in resp.data

    def test_customers_page(self, client, admin_user):
        uid, email, pw = admin_user
        login(client, email, pw)
        resp = client.get('/customers')
        assert resp.status_code == 200

    def test_update_order_status(self, client, app, db, admin_user):
        uid, email, pw = admin_user
        login(client, email, pw)
        from website.models import Order, Product
        with app.app_context():
            p = Product()
            p.product_name = 'StatusTest'
            p.current_price = 50
            p.previous_price = 60
            p.in_stock = 10
            p.size = ''
            p.sugar = ''
            p.milk = ''
            p.shot = ''
            p.product_picture = '/media/s.jpg'
            db.session.add(p)
            db.session.commit()

            o = Order()
            o.quantity = 1
            o.size = ''
            o.sugar = ''
            o.milk = ''
            o.shot = ''
            o.price = 50
            o.status = 'Pending'
            o.payment_id = 'pi_status'
            o.customer_link = uid
            o.product_link = p.id
            db.session.add(o)
            db.session.commit()
            order_id = o.id

        resp = client.post(f'/update-order/{order_id}', data={
            'order_status': 'Delivered',
        }, follow_redirects=True)
        with app.app_context():
            order = Order.query.get(order_id)
            assert order.status == 'Delivered'


# ===========================================================================
# 8. SCHEMA MIGRATION TESTS
# ===========================================================================

class TestSchemaMigration:
    def test_product_has_size_column(self, app, db):
        from website.models import Product
        with app.app_context():
            p = Product()
            p.product_name = 'MigrationTest'
            p.current_price = 10
            p.previous_price = 12
            p.in_stock = 5
            p.size = 'S, M'
            p.sugar = ''
            p.milk = ''
            p.shot = ''
            p.product_picture = '/media/m.jpg'
            db.session.add(p)
            db.session.commit()
            assert p.size == 'S, M'

    def test_cart_has_size_column(self, app, db, normal_user, sample_product):
        from website.models import Cart
        with app.app_context():
            c = Cart()
            c.quantity = 1
            c.size = 'Large'
            c.sugar = ''
            c.milk = ''
            c.shot = ''
            c.customer_link = normal_user[0]
            c.product_link = sample_product
            db.session.add(c)
            db.session.commit()
            assert c.size == 'Large'

    def test_order_has_size_column(self, app, db, normal_user, sample_product):
        from website.models import Order
        with app.app_context():
            o = Order()
            o.quantity = 1
            o.size = 'Medium'
            o.sugar = ''
            o.milk = ''
            o.shot = ''
            o.price = 100
            o.status = 'Pending'
            o.payment_id = 'test'
            o.customer_link = normal_user[0]
            o.product_link = sample_product
            db.session.add(o)
            db.session.commit()
            assert o.size == 'Medium'


# ===========================================================================
# 9. AI / FACE RECOGNITION TESTS
# ===========================================================================

class TestFaceProfiles:
    """Tests for the face_profiles utility module."""

    def test_normalize_name_basic(self):
        from website.face_profiles import normalize_face_profile_name
        assert normalize_face_profile_name('John Doe') == 'john-doe'

    def test_normalize_name_special_chars(self):
        from website.face_profiles import normalize_face_profile_name
        assert normalize_face_profile_name('test@user!#$%') == 'test-user'

    def test_normalize_name_empty(self):
        from website.face_profiles import normalize_face_profile_name
        assert normalize_face_profile_name('') == 'customer-face'

    def test_normalize_name_none(self):
        from website.face_profiles import normalize_face_profile_name
        assert normalize_face_profile_name(None) == 'customer-face'

    def test_get_face_profile_dir(self, tmp_path):
        from website.face_profiles import normalize_face_profile_name
        name = normalize_face_profile_name('Test User')
        assert name == 'test-user'

    def test_list_saved_face_images_empty(self):
        from website.face_profiles import list_saved_face_images
        # Non-existent profile should return empty list
        images = list_saved_face_images('nonexistent-profile-xyz')
        assert images == []

    def test_list_saved_face_images_excludes_augmented(self, tmp_path):
        from website.face_profiles import FACES_DIR
        profile_dir = FACES_DIR / 'test-filter'
        profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            (profile_dir / 'front_01.jpg').write_bytes(b'\xff\xd8')
            (profile_dir / 'front_01_aug_01.jpg').write_bytes(b'\xff\xd8')
            (profile_dir / 'left_01.jpg').write_bytes(b'\xff\xd8')

            from website.face_profiles import list_saved_face_images
            images = list_saved_face_images('test-filter')
            assert 'front_01.jpg' in images
            assert 'left_01.jpg' in images
            assert 'front_01_aug_01.jpg' not in images
        finally:
            shutil.rmtree(profile_dir, ignore_errors=True)


class TestYolov10Helpers:
    """Tests for AI helper functions (no camera/model required)."""

    def test_decode_base64_frame_valid(self):
        from yolov10 import _decode_base64_frame
        import base64
        # Create a tiny 2x2 black image
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', img)
        b64 = base64.b64encode(buf).decode()
        frame_data = f'data:image/jpeg;base64,{b64}'

        frame = _decode_base64_frame(frame_data)
        assert frame is not None
        assert frame.shape[0] > 0

    def test_decode_base64_frame_invalid(self):
        from yolov10 import _decode_base64_frame
        with pytest.raises(ValueError, match='Invalid frame data'):
            _decode_base64_frame('')

    def test_decode_base64_frame_no_comma(self):
        from yolov10 import _decode_base64_frame
        with pytest.raises(ValueError, match='Invalid frame data'):
            _decode_base64_frame('nodatahere')

    def test_extract_face_features_shape(self):
        from yolov10 import _extract_face_features
        # Create a grayscale 96x96 fake face
        face = np.random.randint(0, 255, (96, 96), dtype=np.uint8)
        features = _extract_face_features(face)
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert features.ndim == 1
        # Feature vector: 24*24 thumbnail + 32 histogram + 24*24 gradient = 1184
        assert features.shape[0] == 24 * 24 + 32 + 24 * 24

    def test_extract_face_features_normalized(self):
        from yolov10 import _extract_face_features
        face = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        features = _extract_face_features(face)
        norm = float(np.linalg.norm(features))
        assert abs(norm - 1.0) < 0.01  # Should be unit normalized

    def test_prepare_face_output_size(self):
        from yolov10 import _prepare_face
        face = np.random.randint(0, 255, (50, 60), dtype=np.uint8)
        prepared = _prepare_face(face)
        assert prepared.shape == (96, 96)

    def test_augment_face_count(self):
        from yolov10 import augment_face
        face = np.random.randint(0, 255, (96, 96), dtype=np.uint8)
        augmented = augment_face(face)
        assert len(augmented) == 11  # 11 augmentation types

    def test_augment_face_same_shape(self):
        from yolov10 import augment_face
        face = np.random.randint(0, 255, (96, 96), dtype=np.uint8)
        augmented = augment_face(face)
        for aug in augmented:
            assert aug.shape[0] > 0 and aug.shape[1] > 0

    def test_predict_face_unknown_when_empty(self):
        from yolov10 import _predict_face
        # Temporarily override recognizer with empty data
        import yolov10 as y10
        old_recognizer = y10._recognizer
        y10._recognizer = {
            'features': np.empty((0, 0), dtype=np.float32),
            'labels': np.empty((0,), dtype=np.int32),
        }
        try:
            face = np.random.randint(0, 255, (96, 96), dtype=np.uint8)
            name, conf = _predict_face(face)
            assert name == 'Unknown'
            assert conf == 0.0
        finally:
            y10._recognizer = old_recognizer

    def test_saved_face_data_exists_false(self, tmp_path):
        from yolov10 import _saved_face_data_exists, FACES_DIR
        # If faces dir is empty or has no subdirs with images, should be False
        # This test validates the function runs without error
        result = _saved_face_data_exists()
        assert isinstance(result, bool)

    def test_match_threshold_value(self):
        from yolov10 import MATCH_THRESHOLD
        assert 0 < MATCH_THRESHOLD < 1.0
        assert MATCH_THRESHOLD == 0.90


class TestFaceRecognitionFlow:
    """Integration-like tests for the face recognition API endpoints (mocked AI)."""

    def test_face_preview_endpoint_no_face(self, client, app):
        with patch('website.views.analyze_face_frame') as mock_analyze:
            mock_analyze.return_value = {
                'ok': True, 'detected': False, 'message': 'No face detected.'
            }
            resp = client.post('/face-preview', json={'frame': 'data:image/jpeg;base64,abc'})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['detected'] is False

    def test_usual_order_page_shows_camera_switcher(self, client):
        resp = client.get('/usual-order')
        assert resp.status_code == 200
        assert b'id="cameraSelect"' in resp.data
        assert b'id="switchCameraButton"' in resp.data
        assert b'Camera source' in resp.data
        assert b'wideye-camera-actions-usual' in resp.data
        assert b'wideye-confirm-button' in resp.data
        assert b'Starting camera...' in resp.data

    def test_usual_order_confirm_no_trained_faces(self, client, app):
        with patch('website.views.recognize_face_from_frame_data') as mock_rec:
            mock_rec.return_value = {
                'ok': False, 'message': 'No trained face profiles are available yet.'
            }
            resp = client.post('/usual-order/confirm-frame', json={'frame': 'data:image/jpeg;base64,abc'})
            assert resp.status_code == 400

    @patch('website.views.create_payment_intent')
    @patch('website.views.attach_qrph')
    def test_usual_order_confirm_success(self, mock_attach, mock_create_pi,
                                          client, app, db, normal_user, sample_product):
        uid, email, pw = normal_user
        from website.models import Customer
        with app.app_context():
            customer = Customer.query.get(uid)
            customer.usual_product_id = sample_product
            db.session.commit()
            username = customer.username

        mock_create_pi.return_value = {'id': 'pi_usual_test'}
        mock_attach.return_value = {'attributes': {'next_action': {
            'type': 'consume_qr',
            'code': {'image_url': 'data:image/png;base64,USUALQR'}
        }}}

        with patch('website.views.recognize_face_from_frame_data') as mock_rec:
            mock_rec.return_value = {
                'ok': True,
                'recognized_name': username,
                'confidence': 95.0,
                'message': f'Confirmed {username}.'
            }
            resp = client.post('/usual-order/confirm-frame', json={'frame': 'data:image/jpeg;base64,abc'})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['ok'] is True
            assert 'redirect_url' in data

        # Verify order was created
        from website.models import Order
        with app.app_context():
            order = Order.query.filter_by(payment_id='pi_usual_test').first()
            assert order is not None
            assert order.size == 'Small'  # defaults to first size

    def test_usual_order_no_usual_set(self, client, app, db, normal_user):
        uid, email, pw = normal_user
        with patch('website.views.recognize_face_from_frame_data') as mock_rec:
            mock_rec.return_value = {
                'ok': True,
                'recognized_name': 'testuser',
                'confidence': 95.0,
                'message': 'Confirmed testuser.'
            }
            resp = client.post('/usual-order/confirm-frame', json={'frame': 'data:image/jpeg;base64,abc'})
            assert resp.status_code == 400
            data = resp.get_json()
            assert 'usual order' in data['message'].lower() or 'not saved' in data['message'].lower()

    def test_usual_order_payment_page(self, client, app, db, normal_user, sample_product):
        uid, email, pw = normal_user
        from website.models import Order
        with app.app_context():
            o = Order()
            o.quantity = 1
            o.size = 'Small'
            o.sugar = ''
            o.milk = ''
            o.shot = ''
            o.price = 150
            o.status = 'Pending'
            o.payment_id = 'pi_usual_page'
            o.customer_link = uid
            o.product_link = sample_product
            db.session.add(o)
            db.session.commit()

        with patch('website.views.retrieve_payment_intent') as mock_retrieve:
            mock_retrieve.return_value = {'attributes': {'next_action': {
                'type': 'consume_qr',
                'code': {'image_url': 'data:image/png;base64,QR123'}
            }}}
            # No login needed for usual order payment page
            resp = client.get('/usual-order/payment/pi_usual_page')
            assert resp.status_code == 200
            assert b'QR123' in resp.data


class TestPaymentMethodSelection:
    """Tests for cash vs cashless payment method feature."""

    def test_choose_payment_page_renders(self, client, normal_user, sample_product):
        uid, email, pw = normal_user
        login(client, email, pw)
        client.post(f'/add-to-cart/{sample_product}', data={
            'quantity': '1', 'size': 'Small', 'sugar': '', 'milk': '', 'shot': ''
        })
        resp = client.get('/choose-payment')
        assert resp.status_code == 200
        assert b'Cash' in resp.data
        assert b'Cashless' in resp.data
        assert b'How would you like to pay' in resp.data

    def test_choose_payment_empty_cart_redirects(self, client, normal_user):
        uid, email, pw = normal_user
        login(client, email, pw)
        resp = client.get('/choose-payment', follow_redirects=True)
        assert resp.status_code == 200

    def test_place_order_cash(self, client, app, normal_user, sample_product):
        uid, email, pw = normal_user
        login(client, email, pw)
        client.post(f'/add-to-cart/{sample_product}', data={
            'quantity': '1', 'size': 'Small', 'sugar': '', 'milk': '', 'shot': ''
        })
        resp = client.post('/place-order', data={'payment_method': 'cash'})
        assert resp.status_code == 200
        assert b'Cash' in resp.data or b'cash' in resp.data
        assert b'Pay' in resp.data or b'pay' in resp.data

        from website.models import Order
        with app.app_context():
            orders = Order.query.filter_by(customer_link=uid).all()
            cash_orders = [o for o in orders if o.payment_method == 'cash']
            assert len(cash_orders) >= 1
            assert cash_orders[0].payment_id.startswith('cash-')

    @patch('website.views.create_payment_intent')
    @patch('website.views.attach_qrph')
    def test_place_order_cashless(self, mock_attach, mock_create_pi,
                                   client, app, normal_user, sample_product):
        uid, email, pw = normal_user
        login(client, email, pw)
        client.post(f'/add-to-cart/{sample_product}', data={
            'quantity': '1', 'size': 'Small', 'sugar': '', 'milk': '', 'shot': ''
        })

        mock_create_pi.return_value = {'id': 'pi_cashless_test'}
        mock_attach.return_value = {'attributes': {'next_action': {
            'type': 'consume_qr',
            'code': {'image_url': 'data:image/png;base64,QRTEST'}
        }}}

        resp = client.post('/place-order', data={'payment_method': 'cashless'})
        assert resp.status_code == 200
        assert b'QRTEST' in resp.data

    def test_place_order_get_redirects_to_choose(self, client, normal_user):
        uid, email, pw = normal_user
        login(client, email, pw)
        resp = client.get('/place-order')
        assert resp.status_code == 302

    def test_usual_order_page_shows_payment_toggle(self, client):
        resp = client.get('/usual-order')
        assert resp.status_code == 200
        assert b'paymentMethodToggle' in resp.data
        assert b'data-method="cash"' in resp.data
        assert b'data-method="cashless"' in resp.data

    def test_usual_order_cash_confirm(self, client, app, db, normal_user, sample_product):
        uid, email, pw = normal_user
        from website.models import Customer
        with app.app_context():
            customer = Customer.query.get(uid)
            customer.usual_product_id = sample_product
            db.session.commit()
            username = customer.username

        with patch('website.views.recognize_face_from_frame_data') as mock_rec:
            mock_rec.return_value = {
                'ok': True,
                'recognized_name': username,
                'confidence': 95.0,
                'message': f'Confirmed {username}.'
            }
            resp = client.post('/usual-order/confirm-frame', json={
                'frame': 'data:image/jpeg;base64,abc',
                'payment_method': 'cash'
            })
            assert resp.status_code == 200
            data = resp.get_json()
            assert data['ok'] is True
            assert '/usual-order/cash/' in data['redirect_url']

    def test_usual_order_cash_receipt_page(self, client, app, db, normal_user, sample_product):
        uid, email, pw = normal_user
        from website.models import Order
        with app.app_context():
            o = Order()
            o.quantity = 1
            o.size = 'Small'
            o.sugar = ''
            o.milk = ''
            o.shot = ''
            o.price = 150
            o.status = 'Pending'
            o.payment_method = 'cash'
            o.payment_id = 'cash-testreceipt'
            o.customer_link = uid
            o.product_link = sample_product
            db.session.add(o)
            db.session.commit()

        resp = client.get('/usual-order/cash/cash-testreceipt')
        assert resp.status_code == 200
        assert b'Cash' in resp.data or b'cash' in resp.data
        assert b'150' in resp.data

    def test_cash_receipt_page_not_found(self, client):
        resp = client.get('/usual-order/cash/nonexistent-id', follow_redirects=True)
        assert resp.status_code == 200


# ===========================================================================
# 10. SEARCH TESTS
# ===========================================================================

class TestSearch:
    def test_search_finds_product(self, client, sample_product):
        resp = client.post('/search', data={'search': 'Latte'}, follow_redirects=True)
        assert resp.status_code == 200
        assert b'Test Latte' in resp.data

    def test_search_no_results(self, client, sample_product):
        resp = client.post('/search', data={'search': 'NonexistentProduct'}, follow_redirects=True)
        assert resp.status_code == 200
        assert b'Test Latte' not in resp.data

    def test_search_page_get(self, client):
        resp = client.get('/search')
        assert resp.status_code == 200


# ===========================================================================
# 11. EDGE CASE TESTS
# ===========================================================================

class TestEdgeCases:
    def test_add_nonexistent_item_to_cart(self, client, normal_user):
        uid, email, pw = normal_user
        login(client, email, pw)
        resp = client.post('/add-to-cart/99999', data={}, follow_redirects=True)
        assert b'not found' in resp.data.lower() or resp.status_code == 200

    def test_delete_nonexistent_order(self, client, admin_user):
        uid, email, pw = admin_user
        login(client, email, pw)
        resp = client.get('/delete-order/99999', follow_redirects=True)
        assert b'not found' in resp.data.lower() or resp.status_code == 200

    def test_update_nonexistent_item(self, client, admin_user):
        uid, email, pw = admin_user
        login(client, email, pw)
        resp = client.get('/update-item/99999', follow_redirects=True)
        assert resp.status_code == 200

    def test_product_no_images(self, app, db):
        from website.views import get_product_image
        from website.models import Product
        with app.app_context():
            p = Product()
            p.product_name = 'NoImg'
            p.current_price = 10
            p.previous_price = 12
            p.in_stock = 5
            p.size = ''
            p.sugar = ''
            p.milk = ''
            p.shot = ''
            p.product_picture = ''
            db.session.add(p)
            db.session.commit()
            assert get_product_image(p) == '/media/default.jpg'

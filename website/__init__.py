import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from sqlalchemy import inspect, text

load_dotenv()


db = SQLAlchemy()
BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = BASE_DIR / 'instance'
MEDIA_DIR = BASE_DIR / 'media'
DB_NAME = 'database.sqlite3'


def create_database():
    db.create_all()
    print('Database Created')


def _parse_csv_options(value):
    if not value:
        return []
    return [opt.strip() for opt in str(value).split(',') if opt.strip()]


def sync_customer_schema():
    inspector = inspect(db.engine)

    if not inspector.has_table('customer'):
        return

    existing_columns = {column['name'] for column in inspector.get_columns('customer')}
    statements = []

    if 'face_profile_name' not in existing_columns:
        statements.append('ALTER TABLE customer ADD COLUMN face_profile_name VARCHAR(100)')

    if 'is_admin' not in existing_columns:
        statements.append('ALTER TABLE customer ADD COLUMN is_admin BOOLEAN DEFAULT 0 NOT NULL')

    if 'usual_product_id' not in existing_columns:
        statements.append('ALTER TABLE customer ADD COLUMN usual_product_id INTEGER')

    with db.engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

        connection.execute(text(
            'UPDATE customer SET is_admin = COALESCE(is_admin, 0)'
        ))

        connection.execute(text(
            'UPDATE customer SET is_admin = 1 '
            'WHERE id = (SELECT MIN(id) FROM customer) '
            'AND NOT EXISTS (SELECT 1 FROM customer WHERE COALESCE(is_admin, 0) = 1)'
        ))

        connection.execute(text(
            'UPDATE customer SET face_profile_name = username '
            'WHERE face_profile_name IS NULL OR TRIM(face_profile_name) = ""'
        ))


def sync_product_schema():
    inspector = inspect(db.engine)

    if not inspector.has_table('product'):
        return

    existing_columns = {column['name'] for column in inspector.get_columns('product')}
    statements = []

    if 'sugar' not in existing_columns:
        statements.append("ALTER TABLE product ADD COLUMN sugar VARCHAR(500) DEFAULT '' NOT NULL")

    if 'milk' not in existing_columns:
        statements.append("ALTER TABLE product ADD COLUMN milk VARCHAR(500) DEFAULT '' NOT NULL")

    if 'shot' not in existing_columns:
        statements.append("ALTER TABLE product ADD COLUMN shot VARCHAR(500) DEFAULT '' NOT NULL")

    if 'size' not in existing_columns:
        statements.append("ALTER TABLE product ADD COLUMN size VARCHAR(500) DEFAULT '' NOT NULL")

    with db.engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

        # Migrate old integer values to comma-separated option strings
        for col in ('sugar', 'milk', 'shot'):
            connection.execute(text(
                f"UPDATE product SET {col} = '' WHERE {col} IS NULL OR TRIM(CAST({col} AS TEXT)) = '' OR TRIM(CAST({col} AS TEXT)) = '0'"
            ))


def sync_cart_schema():
    inspector = inspect(db.engine)

    if not inspector.has_table('cart'):
        return

    existing_columns = {column['name'] for column in inspector.get_columns('cart')}
    statements = []

    if 'sugar' not in existing_columns:
        statements.append("ALTER TABLE cart ADD COLUMN sugar VARCHAR(100) DEFAULT '' NOT NULL")

    if 'milk' not in existing_columns:
        statements.append("ALTER TABLE cart ADD COLUMN milk VARCHAR(100) DEFAULT '' NOT NULL")

    if 'shot' not in existing_columns:
        statements.append("ALTER TABLE cart ADD COLUMN shot VARCHAR(100) DEFAULT '' NOT NULL")

    if 'size' not in existing_columns:
        statements.append("ALTER TABLE cart ADD COLUMN size VARCHAR(100) DEFAULT '' NOT NULL")

    with db.engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

        for col in ('sugar', 'milk', 'shot'):
            connection.execute(text(
                f"UPDATE cart SET {col} = '' WHERE {col} IS NULL OR TRIM(CAST({col} AS TEXT)) = '' OR TRIM(CAST({col} AS TEXT)) = '0'"
            ))


def sync_order_schema():
    inspector = inspect(db.engine)

    if not inspector.has_table('order'):
        return

    existing_columns = {column['name'] for column in inspector.get_columns('order')}
    statements = []

    if 'payment_id' not in existing_columns:
        statements.append('ALTER TABLE "order" ADD COLUMN payment_id VARCHAR(1000) DEFAULT \'legacy-order\' NOT NULL')

    if 'sugar' not in existing_columns:
        statements.append("ALTER TABLE \"order\" ADD COLUMN sugar VARCHAR(100) DEFAULT '' NOT NULL")

    if 'milk' not in existing_columns:
        statements.append("ALTER TABLE \"order\" ADD COLUMN milk VARCHAR(100) DEFAULT '' NOT NULL")

    if 'shot' not in existing_columns:
        statements.append("ALTER TABLE \"order\" ADD COLUMN shot VARCHAR(100) DEFAULT '' NOT NULL")

    if 'size' not in existing_columns:
        statements.append("ALTER TABLE \"order\" ADD COLUMN size VARCHAR(100) DEFAULT '' NOT NULL")

    if 'payment_method' not in existing_columns:
        statements.append("ALTER TABLE \"order\" ADD COLUMN payment_method VARCHAR(50) DEFAULT 'cashless' NOT NULL")

    if 'date_placed' not in existing_columns:
        statements.append("ALTER TABLE \"order\" ADD COLUMN date_placed DATETIME DEFAULT NULL")

    with db.engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

        connection.execute(text(
            'UPDATE "order" SET payment_id = COALESCE(NULLIF(TRIM(payment_id), \'\'), \'legacy-order\')'
        ))

        connection.execute(text(
            "UPDATE \"order\" SET date_placed = datetime('now') WHERE date_placed IS NULL"
        ))

        for col in ('sugar', 'milk', 'shot'):
            connection.execute(text(
                f'UPDATE "order" SET {col} = \'\' WHERE {col} IS NULL OR TRIM(CAST({col} AS TEXT)) = \'\' OR TRIM(CAST({col} AS TEXT)) = \'0\''
            ))


def sync_usual_order_schema():
    inspector = inspect(db.engine)

    if not inspector.has_table('usual_order_item'):
        return

    from .models import Customer, UsualOrderItem

    migrated = False
    customers = Customer.query.filter(Customer.usual_product_id.isnot(None)).all()

    for customer in customers:
        if customer.usual_items:
            continue

        if customer.usual_product is None:
            customer.usual_product_id = None
            migrated = True
            continue

        size_options = _parse_csv_options(customer.usual_product.size)
        db.session.add(UsualOrderItem(
            customer_link=customer.id,
            product_link=customer.usual_product.id,
            quantity=1,
            size=size_options[0] if size_options else '',
            sugar='',
            milk='',
            shot='',
        ))
        migrated = True

    if migrated:
        db.session.commit()


def cleanup_products_without_pictures():
    from .models import Product, Cart, Order, Customer, UsualOrderItem

    changed = False
    products = Product.query.all()

    for product in products:
        if _parse_csv_options(product.product_picture):
            continue

        has_orders = Order.query.filter_by(product_link=product.id).first() is not None
        if has_orders:
            product.product_picture = '/media/default.jpg'
            changed = True
            continue

        Cart.query.filter_by(product_link=product.id).delete(synchronize_session=False)
        UsualOrderItem.query.filter_by(product_link=product.id).delete(synchronize_session=False)
        Customer.query.filter_by(usual_product_id=product.id).update(
            {'usual_product_id': None},
            synchronize_session=False
        )
        db.session.delete(product)
        changed = True

    if changed:
        db.session.commit()


def create_app(test_config=None):
    app = Flask(__name__)
    INSTANCE_DIR.mkdir(exist_ok=True)
    MEDIA_DIR.mkdir(exist_ok=True)

    default_database_uri = f'sqlite:///{(INSTANCE_DIR / DB_NAME).as_posix()}'

    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'hbnwdvbn ajnbsjn ahe')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', default_database_uri)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MEDIA_DIR'] = str(MEDIA_DIR)

    if test_config:
        app.config.update(test_config)

    db.init_app(app)

    @app.errorhandler(404)
    def page_not_found(error):
        return render_template('404.html')

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    @login_manager.user_loader
    def load_user(id):
        return Customer.query.get(int(id))

    from .views import views
    from .auth import auth 
    from .admin import admin
    from .models import Customer, Cart, Product, Order

    app.register_blueprint(views, url_prefix='/') # localhost:5000/about-us
    app.register_blueprint(auth, url_prefix='/') # localhost:5000/auth/change-password
    app.register_blueprint(admin, url_prefix='/')

    with app.app_context():
        db.create_all()
        sync_customer_schema()
        sync_product_schema()
        sync_cart_schema()
        sync_order_schema()
        sync_usual_order_schema()
        cleanup_products_without_pictures()
        if not app.testing:
            seed_admin_account()

    return app


def seed_admin_account():
    """Create a default admin account if one doesn't already exist."""
    from .models import Customer

    if Customer.query.filter_by(email='admin@wideye.local').first():
        return

    admin = Customer()
    admin.email = 'admin@wideye.local'
    admin.username = 'admin'
    admin.password = 'admin1'
    admin.is_admin = True
    admin.face_profile_name = 'admin'

    db.session.add(admin)
    db.session.commit()
    print('✔ Default admin account created (admin / admin1)')


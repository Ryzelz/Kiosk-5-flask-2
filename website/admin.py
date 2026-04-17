from pathlib import Path

from flask import Blueprint, render_template, flash, send_from_directory, redirect, current_app, request, jsonify
from flask_login import login_required, current_user
from sqlalchemy import func
from .forms import ShopItemsForm, UpdateShopItemsForm, OrderForm, AdminCustomerUpdateForm
from werkzeug.utils import secure_filename
from .models import Product, Order, Customer
from .views import parse_options, get_product_image
from . import db
import time


admin = Blueprint('admin', __name__)


def get_media_dir():
    media_dir = Path(current_app.config['MEDIA_DIR'])
    media_dir.mkdir(exist_ok=True)
    return media_dir


def build_media_url(file_name):
    return f'/media/{file_name}'


def build_media_path(file_name):
    return get_media_dir() / file_name


def resolve_updated_value(value, fallback):
    if value is None:
        return fallback

    return value


def current_user_is_admin():
    return current_user.is_authenticated and bool(getattr(current_user, 'is_admin', False))


def render_customer_update_page(customer, form):
    order_stats = get_customer_order_stats(customer.id)
    return render_template('update_customer.html', customer=customer, form=form, order_stats=order_stats)


def get_customer_order_stats(customer_id):
    """Return per-product order totals for a customer, sorted by total quantity desc."""
    rows = (
        db.session.query(
            Order.product_link,
            func.sum(Order.quantity).label('total_qty'),
            func.count(Order.id).label('order_count'),
            func.max(Order.date_placed).label('last_ordered'),
        )
        .filter(Order.customer_link == customer_id)
        .group_by(Order.product_link)
        .order_by(func.sum(Order.quantity).desc())
        .all()
    )

    stats = []
    for row in rows:
        product = Product.query.get(row.product_link)
        if product is None:
            continue
        stats.append({
            'product': product,
            'total_qty': int(row.total_qty),
            'order_count': int(row.order_count),
            'last_ordered': row.last_ordered,
        })
    return stats


@admin.route('/media/<path:filename>')
def get_image(filename):
    return send_from_directory(str(get_media_dir()), filename)


@admin.route('/add-shop-items', methods=['GET', 'POST'])
@login_required
def add_shop_items():
    if current_user_is_admin():
        form = ShopItemsForm()

        if request.method == 'POST':
            # Validate non-file fields manually since MultipleFileField
            # can interfere with form.validate_on_submit()
            product_name = form.product_name.data
            current_price = form.current_price.data
            previous_price = form.previous_price.data
            in_stock = form.in_stock.data

            if not product_name:
                flash('Product name is required.')
                return render_template('add_shop_items.html', form=form)
            if current_price is None or previous_price is None:
                flash('Prices are required.')
                return render_template('add_shop_items.html', form=form)
            if in_stock is None:
                flash('Stock quantity is required.')
                return render_template('add_shop_items.html', form=form)

            size = form.size.data or ''
            sugar = form.sugar.data or ''
            milk = form.milk.data or ''
            shot = form.shot.data or ''
            flash_sale = form.flash_sale.data

            files = request.files.getlist('product_picture')
            file_urls = []
            for file in files:
                if file and file.filename:
                    timestamp = str(int(time.time() * 1000))
                    file_name = f"{timestamp}_{secure_filename(file.filename)}"
                    file_path = build_media_path(file_name)
                    file.save(str(file_path))
                    file_urls.append(build_media_url(file_name))

            if not file_urls:
                flash('At least one product picture is required.')
                return render_template('add_shop_items.html', form=form)

            new_shop_item = Product()
            new_shop_item.product_name = product_name
            new_shop_item.current_price = current_price
            new_shop_item.previous_price = previous_price
            new_shop_item.in_stock = in_stock
            new_shop_item.size = size
            new_shop_item.sugar = sugar
            new_shop_item.milk = milk
            new_shop_item.shot = shot
            new_shop_item.flash_sale = flash_sale
            new_shop_item.product_picture = ', '.join(file_urls)

            try:
                db.session.add(new_shop_item)
                db.session.commit()
                flash(f'{product_name} added Successfully')
                print('Product Added')
                return render_template('add_shop_items.html', form=form)
            except Exception as e:
                print(e)
                flash('Product Not Added!!')

        return render_template('add_shop_items.html', form=form)

    return render_template('404.html')


@admin.route('/shop-items', methods=['GET', 'POST'])
@login_required
def shop_items():
    if current_user_is_admin():
        items = Product.query.order_by(Product.date_added).all()
        return render_template('shop_items.html', items=items,
                               parse_options=parse_options, get_product_image=get_product_image)
    return render_template('404.html')


@admin.route('/update-item/<int:item_id>', methods=['GET', 'POST'])
@login_required
def update_item(item_id):
    if current_user_is_admin():
        form = UpdateShopItemsForm()

        item_to_update = Product.query.get(item_id)

        if item_to_update is None:
            flash('Item not found.')
            return redirect('/shop-items')

        if request.method == 'GET':
            form.product_name.data = item_to_update.product_name
            form.previous_price.data = item_to_update.previous_price
            form.current_price.data = item_to_update.current_price
            form.in_stock.data = item_to_update.in_stock
            form.size.data = item_to_update.size
            form.sugar.data = item_to_update.sugar
            form.milk.data = item_to_update.milk
            form.shot.data = item_to_update.shot
            form.flash_sale.data = item_to_update.flash_sale

        if request.method == 'POST':
            product_name = resolve_updated_value(form.product_name.data, item_to_update.product_name)
            current_price = resolve_updated_value(form.current_price.data, item_to_update.current_price)
            previous_price = resolve_updated_value(form.previous_price.data, item_to_update.previous_price)
            in_stock = resolve_updated_value(form.in_stock.data, item_to_update.in_stock)
            size = resolve_updated_value(form.size.data, item_to_update.size)
            sugar = resolve_updated_value(form.sugar.data, item_to_update.sugar)
            milk = resolve_updated_value(form.milk.data, item_to_update.milk)
            shot = resolve_updated_value(form.shot.data, item_to_update.shot)
            flash_sale = form.flash_sale.data

            files = request.files.getlist('product_picture')
            product_picture = item_to_update.product_picture

            has_new_files = any(f and f.filename for f in files)
            if has_new_files:
                file_urls = []
                for file in files:
                    if file and file.filename:
                        timestamp = str(int(time.time() * 1000))
                        file_name = f"{timestamp}_{secure_filename(file.filename)}"
                        file_path = build_media_path(file_name)
                        file.save(str(file_path))
                        file_urls.append(build_media_url(file_name))
                product_picture = ', '.join(file_urls)

            try:
                Product.query.filter_by(id=item_id).update(dict(product_name=product_name,
                                                                current_price=current_price,
                                                                previous_price=previous_price,
                                                                in_stock=in_stock,
                                                                size=size,
                                                                sugar=sugar,
                                                                milk=milk,
                                                                shot=shot,
                                                                flash_sale=flash_sale,
                                                                product_picture=product_picture))

                db.session.commit()
                flash(f'{product_name} updated Successfully')
                print('Product Upadted')
                return redirect('/shop-items')
            except Exception as e:
                print('Product not Upated', e)
                flash('Item Not Updated!!!')

        return render_template('update_item.html', form=form, item=item_to_update,
                               parse_options=parse_options, get_product_image=get_product_image)
    return render_template('404.html')


@admin.route('/delete-item/<int:item_id>', methods=['GET', 'POST'])
@login_required
def delete_item(item_id):
    if current_user_is_admin():
        try:
            item_to_delete = Product.query.get(item_id)
            db.session.delete(item_to_delete)
            db.session.commit()
            flash('One Item deleted')
            return redirect('/shop-items')
        except Exception as e:
            print('Item not deleted', e)
            flash('Item not deleted!!')
        return redirect('/shop-items')

    return render_template('404.html')


@admin.route('/view-orders')
@login_required
def order_view():
    if current_user_is_admin():
        sort = request.args.get('sort', 'desc')
        if sort == 'asc':
            orders = Order.query.order_by(Order.id.asc()).all()
        else:
            orders = Order.query.order_by(Order.id.desc()).all()
        order_form = OrderForm()
        return render_template(
            'view_orders.html',
            orders=orders,
            get_product_image=get_product_image,
            order_form=order_form,
            sort=sort,
        )
    return render_template('404.html')


@admin.route('/update-order/<int:order_id>', methods=['GET', 'POST'])
@login_required
def update_order(order_id):
    if current_user_is_admin():
        form = OrderForm()

        order = Order.query.get(order_id)

        if order is None:
            flash('Order not found.')
            return redirect('/view-orders')

        if not form.is_submitted():
            form.order_status.data = order.status

        if form.validate_on_submit():
            status = form.order_status.data
            order.status = status

            try:
                db.session.commit()
                flash(f'Order {order_id} Updated successfully')
                return redirect('/view-orders')
            except Exception as e:
                print(e)
                flash(f'Order {order_id} not updated')
                return redirect('/view-orders')

        return render_template('order_update.html', form=form, order=order, get_product_image=get_product_image)

    return render_template('404.html')


@admin.route('/delete-order/<int:order_id>', methods=['GET', 'POST'])
@login_required
def delete_order(order_id):
    if current_user_is_admin():
        try:
            order_to_delete = Order.query.get(order_id)
            if order_to_delete is None:
                flash('Order not found.')
                return redirect('/view-orders')
            db.session.delete(order_to_delete)
            db.session.commit()
            flash(f'Order #{order_id} deleted successfully.')
        except Exception as e:
            print('Order not deleted', e)
            flash('Order could not be deleted!')
        return redirect('/view-orders')

    return render_template('404.html')


@admin.route('/customers')
@login_required
def display_customers():
    if current_user_is_admin():
        customers = Customer.query.order_by(Customer.date_joined.desc(), Customer.id.desc()).all()
        # Build a quick top-item lookup for the table
        top_items = {}
        for customer in customers:
            stats = get_customer_order_stats(customer.id)
            top_items[customer.id] = stats[0] if stats else None
        return render_template('customers.html', customers=customers, top_items=top_items)
    return render_template('404.html')


@admin.route('/update-customer/<int:customer_id>', methods=['GET', 'POST'])
@login_required
def update_customer(customer_id):
    if current_user_is_admin():
        customer = Customer.query.get(customer_id)
        if customer is None:
            flash('Customer not found.')
            return redirect('/customers')

        form = AdminCustomerUpdateForm()

        if request.method == 'GET':
            form.email.data = customer.email
            form.username.data = customer.username
            form.is_admin.data = customer.is_admin
            return render_customer_update_page(customer, form)

        if not form.validate_on_submit():
            for field_errors in form.errors.values():
                for error in field_errors:
                    flash(error)
            return render_customer_update_page(customer, form)

        new_email = form.email.data.strip()
        new_username = form.username.data.strip()
        make_admin = bool(form.is_admin.data)

        existing_email = Customer.query.filter(
            Customer.id != customer.id,
            func.lower(Customer.email) == new_email.lower(),
        ).first()
        if existing_email:
            flash('That email is already in use.')
            return render_customer_update_page(customer, form)

        existing_username = Customer.query.filter(
            Customer.id != customer.id,
            func.lower(Customer.username) == new_username.lower(),
        ).first()
        if existing_username:
            flash('That username is already in use.')
            return render_customer_update_page(customer, form)

        if customer.is_admin and not make_admin:
            other_admin = Customer.query.filter(
                Customer.id != customer.id,
                Customer.is_admin.is_(True),
            ).first()
            if other_admin is None:
                flash('You cannot remove admin access from the last admin account.')
                return render_customer_update_page(customer, form)

        try:
            from .auth import sync_face_profile_assets
            sync_face_profile_assets(customer, new_username)
            customer.email = new_email
            customer.username = new_username
            customer.is_admin = make_admin
            db.session.commit()
            flash('Customer updated successfully.')
            return redirect('/customers')
        except Exception as e:
            db.session.rollback()
            print('Customer not updated', e)
            flash(f'Customer not updated — {e}')
            return render_customer_update_page(customer, form)

    return render_template('404.html')


@admin.route('/delete-customer/<int:customer_id>', methods=['GET', 'POST'])
@login_required
def delete_customer(customer_id):
    if current_user_is_admin():
        customer = Customer.query.get(customer_id)
        if customer is None:
            flash('Customer not found.')
            return redirect('/customers')

        if customer.id == current_user.id:
            flash('Use your profile page to manage your own account.')
            return redirect('/customers')

        if customer.is_admin:
            other_admin = Customer.query.filter(
                Customer.id != customer.id,
                Customer.is_admin.is_(True),
            ).first()
            if other_admin is None:
                flash('You cannot delete the last admin account.')
                return redirect('/customers')

        try:
            from .auth import delete_customer_account
            delete_customer_account(customer)
            flash('Customer deleted successfully.')
        except Exception as e:
            db.session.rollback()
            print('Customer not deleted', e)
            flash('Customer could not be deleted!')

        return redirect('/customers')

    return render_template('404.html')


@admin.route('/admin-page')
@login_required
def admin_page():
    if current_user_is_admin():
        return render_template('admin.html')
    return render_template('404.html')


@admin.route('/face-model', methods=['GET', 'POST'])
@login_required
def face_model_settings():
    if not current_user_is_admin():
        return render_template('404.html')

    import yolov10 as face_module

    if request.method == 'POST':
        selected = request.form.get('model')
        if selected and selected in face_module.AVAILABLE_MODELS:
            try:
                face_module.set_active_model_name(selected)
                flash(f'Face detection model switched to {face_module.AVAILABLE_MODELS[selected]["label"]}.')
            except Exception as e:
                flash(f'Could not switch model: {e}')
        else:
            flash('Invalid model selection.')
        return redirect('/face-model')

    model_status = face_module.get_model_status()
    return render_template('face_model.html', model_status=model_status)


@admin.route('/face-model/retrain', methods=['POST'])
@login_required
def retrain_face_model():
    if not current_user_is_admin():
        return jsonify({"ok": False, "error": "Unauthorized"}), 403

    import yolov10 as face_module

    try:
        if face_module.TRAINER_FILE.exists():
            face_module.TRAINER_FILE.unlink()
        face_module.train_faces()
        label_map = face_module._load_label_map(force_reload=True)
        n_people = len(label_map)
        return jsonify({"ok": True, "people": n_people, "names": list(label_map.values())})
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@admin.route('/face-model/evaluate', methods=['POST'])
@login_required
def evaluate_face_model():
    if not current_user_is_admin():
        return jsonify({"ok": False, "error": "Unauthorized"}), 403

    import yolov10 as face_module  # noqa: F811

    data = request.get_json(silent=True) or {}
    model_name = data.get('model') or face_module.get_active_model_name()

    if model_name not in face_module.AVAILABLE_MODELS:
        return jsonify({"ok": False, "error": f"Unknown model '{model_name}'"}), 400

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import evaluate_model as em
        result = em.run_evaluation(model_name)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"ok": False, "error": str(e), "traceback": traceback.format_exc()}), 500


@admin.route('/analytics')
@login_required
def analytics_page():
    if not current_user_is_admin():
        return render_template('404.html')

    revenue_orders = Order.query.filter(Order.status.in_(['Accepted', 'Delivered'])).all()
    total_revenue = sum(o.price * o.quantity for o in revenue_orders)

    pending_count = Order.query.filter_by(status='Pending').count()
    total_visitors = Customer.query.filter_by(is_admin=False).count()
    total_orders = Order.query.count()

    latest_orders = Order.query.order_by(Order.id.desc()).limit(10).all()

    popular_raw = db.session.query(
        Product.product_name,
        func.sum(Order.quantity).label('total_qty'),
    ).join(Order, Order.product_link == Product.id) \
     .group_by(Product.id) \
     .order_by(func.sum(Order.quantity).desc()) \
     .limit(6).all()

    popular_labels = [r.product_name for r in popular_raw]
    popular_values = [int(r.total_qty) for r in popular_raw]

    from datetime import datetime, timedelta
    monthly_labels = []
    monthly_values = []
    now = datetime.utcnow()
    for i in range(5, -1, -1):
        first = (now.replace(day=1) - timedelta(days=i * 28)).replace(day=1)
        if first.month == 12:
            last = first.replace(year=first.year + 1, month=1, day=1)
        else:
            last = first.replace(month=first.month + 1, day=1)
        total = db.session.query(func.sum(Order.price)).filter(
            Order.date_placed >= first,
            Order.date_placed < last,
        ).scalar() or 0
        monthly_labels.append(first.strftime('%b %Y'))
        monthly_values.append(round(total, 2))

    has_demo_data = Customer.query.filter(Customer.email.like('%@demo.com')).first() is not None

    return render_template(
        'analytics.html',
        total_revenue=total_revenue,
        pending_count=pending_count,
        total_visitors=total_visitors,
        total_orders=total_orders,
        latest_orders=latest_orders,
        popular_labels=popular_labels,
        popular_values=popular_values,
        monthly_labels=monthly_labels,
        monthly_values=monthly_values,
        get_product_image=get_product_image,
        has_demo_data=has_demo_data,
    )


@admin.route('/seed-demo-data')
@login_required
def seed_demo_data():
    if not current_user_is_admin():
        return render_template('404.html')

    import random
    from werkzeug.security import generate_password_hash

    demo_products = [
        {'name': 'Caramel Latte',        'price': 145, 'prev': 160},
        {'name': 'Matcha Frappe',         'price': 155, 'prev': 170},
        {'name': 'Classic Espresso',      'price': 95,  'prev': 110},
        {'name': 'Hazelnut Cappuccino',   'price': 135, 'prev': 150},
        {'name': 'Vanilla Cold Brew',     'price': 165, 'prev': 180},
    ]
    for dp in demo_products:
        if not Product.query.filter_by(product_name=dp['name']).first():
            p = Product()
            p.product_name    = dp['name']
            p.current_price   = dp['price']
            p.previous_price  = dp['prev']
            p.in_stock        = 50
            p.size            = 'Small, Medium, Large'
            p.sugar           = 'Low, Regular, Extra'
            p.milk            = 'Whole, Oat, Almond'
            p.shot            = 'Single, Double'
            p.product_picture = '/media/default.jpg'
            p.flash_sale      = False
            db.session.add(p)
    db.session.commit()

    demo_customers = [
        ('juan@demo.com',   'juandelacruz'),
        ('maria@demo.com',  'mariasantos'),
        ('pedro@demo.com',  'pedroreyes'),
        ('ana@demo.com',    'anagonzales'),
        ('carlos@demo.com', 'carlosmendoza'),
    ]
    for email, uname in demo_customers:
        if not Customer.query.filter_by(email=email).first():
            c = Customer()
            c.email             = email
            c.username          = uname
            c.password_hash     = generate_password_hash('demo1234')
            c.is_admin          = False
            c.face_profile_name = uname
            db.session.add(c)
    db.session.commit()

    products  = Product.query.filter(Product.product_name.in_(
        [d['name'] for d in demo_products])).all()
    customers = Customer.query.filter(Customer.email.in_(
        [e for e, _ in demo_customers])).all()

    statuses = ['Accepted', 'Accepted', 'Delivered', 'Delivered',
                'Pending',  'Accepted', 'Canceled',  'Delivered']
    sizes    = ['Small', 'Medium', 'Large']
    from datetime import datetime, timedelta

    for i in range(30):
        prod = random.choice(products)
        qty  = random.randint(1, 3)
        # spread across last 6 months
        days_ago = random.randint(0, 180)
        order_date = datetime.utcnow() - timedelta(days=days_ago)
        o = Order()
        o.quantity       = qty
        o.product_link   = prod.id
        o.customer_link  = random.choice(customers).id
        o.size           = random.choice(sizes)
        o.sugar          = 'Regular'
        o.milk           = 'Whole'
        o.shot           = 'Single'
        o.price          = prod.current_price * qty
        o.status         = random.choice(statuses)
        o.payment_method = random.choice(['cash', 'cashless'])
        o.payment_id     = f'demo-{int(time.time())}-{i}'
        o.date_placed    = order_date
        db.session.add(o)
    db.session.commit()

    flash('Demo data seeded! Reload the analytics page to see it.', 'success')
    return redirect('/analytics')


@admin.route('/clear-demo-data')
@login_required
def clear_demo_data():
    if not current_user_is_admin():
        return render_template('404.html')

    demo_emails = ['juan@demo.com', 'maria@demo.com', 'pedro@demo.com',
                   'ana@demo.com', 'carlos@demo.com']
    demo_product_names = ['Caramel Latte', 'Matcha Frappe', 'Classic Espresso',
                          'Hazelnut Cappuccino', 'Vanilla Cold Brew']

    # Remove demo orders (by payment_id prefix)
    Order.query.filter(Order.payment_id.like('demo-%')).delete(synchronize_session=False)

    # Remove demo customers
    Customer.query.filter(Customer.email.in_(demo_emails)).delete(synchronize_session=False)

    # Remove demo products only if they have no remaining orders
    for name in demo_product_names:
        product = Product.query.filter_by(product_name=name).first()
        if product and Order.query.filter_by(product_link=product.id).count() == 0:
            db.session.delete(product)

    db.session.commit()
    flash('Demo data cleared. Showing live data.', 'success')
    return redirect('/analytics')










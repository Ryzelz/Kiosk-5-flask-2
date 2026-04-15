from pathlib import Path

from flask import Blueprint, render_template, flash, send_from_directory, redirect, current_app, request
from flask_login import login_required, current_user
from .forms import ShopItemsForm, UpdateShopItemsForm, OrderForm
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
        orders = Order.query.all()
        return render_template('view_orders.html', orders=orders, get_product_image=get_product_image)
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
        customers = Customer.query.all()
        return render_template('customers.html', customers=customers)
    return render_template('404.html')


@admin.route('/admin-page')
@login_required
def admin_page():
    if current_user_is_admin():
        return render_template('admin.html')
    return render_template('404.html')










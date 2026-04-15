from flask import Blueprint, render_template, flash, redirect, request, jsonify, url_for
from sqlalchemy import func, or_

from .models import Product, Cart, Order, Customer
from flask_login import login_required, current_user
from . import db
from .paymongo import create_payment_intent, attach_qrph, retrieve_payment_intent


views = Blueprint('views', __name__)


def clamp_option_count(value, max_count=5):
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return 0

    return max(0, min(numeric_value, max_count))


def format_option_summary(item):
    parts = []

    if getattr(item, 'sugar', 0):
        parts.append(f'Sugar: {clamp_option_count(item.sugar)}')

    if getattr(item, 'milk', 0):
        parts.append(f'Milk: {clamp_option_count(item.milk)}')

    if getattr(item, 'shot', 0):
        parts.append(f'Shot: {clamp_option_count(item.shot)}')

    return ', '.join(parts) if parts else 'No add-ons'


def add_product_to_customer_cart(customer_id, product, sugar=0, milk=0, shot=0):
    selected_sugar = clamp_option_count(sugar)
    selected_milk = clamp_option_count(milk)
    selected_shot = clamp_option_count(shot)

    cart_item = Cart.query.filter_by(
        customer_link=customer_id,
        product_link=product.id,
        sugar=selected_sugar,
        milk=selected_milk,
        shot=selected_shot
    ).first()

    if cart_item:
        cart_item.quantity += 1
    else:
        cart_item = Cart()
        cart_item.quantity = 1
        cart_item.sugar = selected_sugar
        cart_item.milk = selected_milk
        cart_item.shot = selected_shot
        cart_item.customer_link = customer_id
        cart_item.product_link = product.id
        db.session.add(cart_item)

    db.session.commit()
    return cart_item


def add_customer_usual_to_cart(customer):
    if customer.usual_product is None:
        return False, f'{customer.username} has not saved a usual order yet.'

    if customer.usual_product.in_stock < 1:
        return False, f'{customer.usual_product.product_name} is currently out of stock.'

    add_product_to_customer_cart(customer.id, customer.usual_product)
    return True, f'Usual item added to cart for {customer.username}: {customer.usual_product.product_name}.'


@views.route('/')
def home():

    items = Product.query.order_by(Product.date_added.desc()).all()

    return render_template('home.html', items=items, format_option_summary=format_option_summary,
                           cart=Cart.query.filter_by(customer_link=current_user.id).all()
                           if current_user.is_authenticated else [])


@views.route('/add-to-cart/<int:item_id>', methods=['GET', 'POST'])
@login_required
def add_to_cart(item_id):
    item_to_add = Product.query.get(item_id)
    if item_to_add is None:
        flash('Item not found')
        return redirect(request.referrer or '/')

    selected_sugar = clamp_option_count(request.values.get('sugar', 0))
    selected_milk = clamp_option_count(request.values.get('milk', 0))
    selected_shot = clamp_option_count(request.values.get('shot', 0))

    if item_to_add.sugar <= 0:
        selected_sugar = 0
    else:
        selected_sugar = min(selected_sugar, clamp_option_count(item_to_add.sugar))

    if item_to_add.milk <= 0:
        selected_milk = 0
    else:
        selected_milk = min(selected_milk, clamp_option_count(item_to_add.milk))

    if item_to_add.shot <= 0:
        selected_shot = 0
    else:
        selected_shot = min(selected_shot, clamp_option_count(item_to_add.shot))

    try:
        cart_item = add_product_to_customer_cart(
            current_user.id,
            item_to_add,
            sugar=selected_sugar,
            milk=selected_milk,
            shot=selected_shot
        )
        flash(f'{cart_item.product.product_name} added to cart')
    except Exception as e:
        print('Item not added to cart', e)
        flash(f'{item_to_add.product_name} has not been added to cart')

    return redirect(request.referrer)


@views.route('/cart')
@login_required
def show_cart():
    cart = Cart.query.filter_by(customer_link=current_user.id).all()
    amount = 0
    for item in cart:
        amount += item.product.current_price * item.quantity

    return render_template('cart.html', cart=cart, amount=amount, total=amount+200,
                           format_option_summary=format_option_summary)


@views.route('/pluscart')
@login_required
def plus_cart():
    if request.method == 'GET':
        cart_id = request.args.get('cart_id')
        cart_item = Cart.query.get(cart_id)
        cart_item.quantity = cart_item.quantity + 1
        db.session.commit()

        cart = Cart.query.filter_by(customer_link=current_user.id).all()

        amount = 0

        for item in cart:
            amount += item.product.current_price * item.quantity

        data = {
            'quantity': cart_item.quantity,
            'amount': amount,
            'total': amount + 200
        }

        return jsonify(data)


@views.route('/minuscart')
@login_required
def minus_cart():
    if request.method == 'GET':
        cart_id = request.args.get('cart_id')
        cart_item = Cart.query.get(cart_id)
        cart_item.quantity = cart_item.quantity - 1
        db.session.commit()

        cart = Cart.query.filter_by(customer_link=current_user.id).all()

        amount = 0

        for item in cart:
            amount += item.product.current_price * item.quantity

        data = {
            'quantity': cart_item.quantity,
            'amount': amount,
            'total': amount + 200 #niggg
        }

        return jsonify(data)


@views.route('removecart')
@login_required
def remove_cart():
    if request.method == 'GET':
        cart_id = request.args.get('cart_id')
        cart_item = Cart.query.get(cart_id)
        db.session.delete(cart_item)
        db.session.commit()

        cart = Cart.query.filter_by(customer_link=current_user.id).all()

        amount = 0

        for item in cart:
            amount += item.product.current_price * item.quantity

        data = {
            'quantity': cart_item.quantity,
            'amount': amount,
            'total': amount + 200
        }

        return jsonify(data)


@views.route('/place-order')
@login_required
def place_order():
    customer_cart = Cart.query.filter_by(customer_link=current_user.id).all()
    if not customer_cart:
        flash('Your cart is Empty')
        return redirect('/')

    try:
        total = 0
        for item in customer_cart:
            total += item.product.current_price * item.quantity

        total_with_shipping = total + 200

        pi = create_payment_intent(total_with_shipping, description='Wideye Kiosk order')
        pi_id = pi['id']

        attached = attach_qrph(pi_id)

        for item in customer_cart:
            new_order = Order()
            new_order.quantity = item.quantity
            new_order.sugar = item.sugar
            new_order.milk = item.milk
            new_order.shot = item.shot
            new_order.price = item.product.current_price
            new_order.status = 'Pending'
            new_order.payment_id = pi_id

            new_order.product_link = item.product_link
            new_order.customer_link = item.customer_link

            db.session.add(new_order)

            product = Product.query.get(item.product_link)
            product.in_stock -= item.quantity

            db.session.delete(item)

        db.session.commit()

        next_action = attached['attributes'].get('next_action') or {}
        qr_image_url = None
        redirect_url = None

        if next_action.get('type') == 'consume_qr':
            qr_image_url = next_action.get('code', {}).get('image_url')
        elif next_action.get('type') == 'redirect':
            redirect_url = next_action.get('redirect', {}).get('url')
        elif next_action.get('type') == 'display_qr_code':
            display_details = next_action.get('display_details', {})
            qr_image_url = display_details.get('qr_image')
            redirect_url = display_details.get('checkout_url')

        return render_template(
            'payment_qr.html',
            payment_intent_id=pi_id,
            qr_image_url=qr_image_url,
            redirect_url=redirect_url,
            total=total_with_shipping,
        )
    except Exception as e:
        db.session.rollback()
        print('PayMongo error:', e)
        flash('Order not placed — payment could not be initiated.')
        return redirect('/')


@views.route('/check-payment/<payment_intent_id>')
@login_required
def check_payment(payment_intent_id):
    try:
        pi = retrieve_payment_intent(payment_intent_id)
        status = pi['attributes']['status']

        if status == 'succeeded':
            Order.query.filter_by(
                payment_id=payment_intent_id,
                customer_link=current_user.id,
            ).update({'status': 'Accepted'})
            db.session.commit()

        return jsonify({'status': status})
    except Exception as e:
        print('Payment check error:', e)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@views.route('/paymongo-webhook', methods=['POST'])
def paymongo_webhook():
    payload = request.get_json(silent=True) or {}
    event_type = payload.get('data', {}).get('attributes', {}).get('type', '')

    if event_type in ('payment.paid', 'payment_intent.succeeded'):
        resource = (
            payload.get('data', {})
            .get('attributes', {})
            .get('data', {})
        )
        pi_id = resource.get('id', '')

        if pi_id:
            Order.query.filter_by(payment_id=pi_id).update({'status': 'Accepted'})
            db.session.commit()

    return jsonify({'success': True}), 200


@views.route('/orders')
@login_required
def order():
    orders = Order.query.filter_by(customer_link=current_user.id).all()
    return render_template('orders.html', orders=orders, format_option_summary=format_option_summary)


@views.route('/usual-order', methods=['GET', 'POST'])
def usual_order():
    if request.method == 'POST':
        return redirect(url_for('views.usual_order'))

    return render_template(
        'face_camera.html',
        page_title='Usual order',
        mode='usual',
        back_url=url_for('views.home'),
        preview_url=url_for('views.preview_face_frame'),
        confirm_url=url_for('views.confirm_usual_order_frame'),
        reset_url='',
        success_redirect=url_for('views.show_cart') if current_user.is_authenticated else url_for('views.home'),
    )


@views.route('/face-preview', methods=['POST'])
def preview_face_frame():
    payload = request.get_json(silent=True) or {}
    frame_data = payload.get('frame')

    try:
        from yolov10 import analyze_face_frame
        result = analyze_face_frame(frame_data)
    except Exception as exc:
        print(exc)
        return jsonify({'ok': False, 'message': 'Live face preview could not be completed.'}), 500

    status_code = 200 if result.get('ok') else 400
    return jsonify(result), status_code


@views.route('/usual-order/confirm-frame', methods=['POST'])
def confirm_usual_order_frame():
    payload = request.get_json(silent=True) or {}
    frame_data = payload.get('frame')

    try:
        from yolov10 import recognize_face_from_frame_data
        result = recognize_face_from_frame_data(frame_data)
    except Exception as exc:
        print(exc)
        return jsonify({'ok': False, 'message': 'The facial confirmation could not be completed.'}), 500

    if not result.get('ok'):
        return jsonify(result), 400

    recognized_name = result['recognized_name']
    normalized_name = recognized_name.strip().lower()
    customer = Customer.query.filter(
        or_(
            func.lower(Customer.face_profile_name) == normalized_name,
            func.lower(Customer.username) == normalized_name
        )
    ).first()

    if customer is None:
        return jsonify({'ok': False, 'message': f'No customer profile matched the detected face: {recognized_name}.'}), 404

    ok, message = add_customer_usual_to_cart(customer)

    if not ok:
        return jsonify({'ok': False, 'message': message}), 400

    redirect_url = url_for('views.home')
    if current_user.is_authenticated and current_user.id == customer.id:
        redirect_url = url_for('views.show_cart')

    return jsonify({
        'ok': True,
        'message': message,
        'recognized_name': recognized_name,
        'redirect_url': redirect_url
    })


@views.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_query = request.form.get('search')
        items = Product.query.filter(Product.product_name.ilike(f'%{search_query}%')).all()
        return render_template('search.html', items=items, cart=Cart.query.filter_by(customer_link=current_user.id).all()
                           if current_user.is_authenticated else [])

    return render_template('search.html')















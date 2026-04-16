from flask import Blueprint, render_template, flash, redirect, request, jsonify, url_for
from sqlalchemy import func, or_
import uuid

from .models import Product, Cart, Order, Customer
from flask_login import login_required, current_user
from . import db
from .paymongo import create_payment_intent, attach_qrph, retrieve_payment_intent


views = Blueprint('views', __name__)


def parse_options(csv_string):
    """Parse a comma-separated options string into a list of trimmed, non-empty tokens."""
    if not csv_string:
        return []
    return [opt.strip() for opt in str(csv_string).split(',') if opt.strip()]


def get_product_image(product, selected_size=''):
    """Return the image URL for a product given a selected size.

    Images in product_picture are comma-separated, position-matched to the
    comma-separated sizes in product.size.  Falls back to the first image.
    """
    images = parse_options(product.product_picture) if product.product_picture else []
    if not images:
        return '/media/default.jpg'

    if not selected_size:
        return images[0]

    sizes = parse_options(product.size)
    if selected_size in sizes:
        idx = sizes.index(selected_size)
        return images[idx] if idx < len(images) else images[-1]

    return images[0]


def normalize_product_selection(product, size='', sugar='', milk='', shot=''):
    selected_size = (size or '').strip()
    selected_sugar = (sugar or '').strip()
    selected_milk = (milk or '').strip()
    selected_shot = (shot or '').strip()

    product_sizes = parse_options(product.size)
    product_sugars = parse_options(product.sugar)
    product_milks = parse_options(product.milk)
    product_shots = parse_options(product.shot)

    if selected_size and selected_size not in product_sizes:
        selected_size = ''
    if product_sizes and not selected_size:
        selected_size = product_sizes[0]
    if selected_sugar and selected_sugar not in product_sugars:
        selected_sugar = ''
    if selected_milk and selected_milk not in product_milks:
        selected_milk = ''
    if selected_shot and selected_shot not in product_shots:
        selected_shot = ''

    return {
        'size': selected_size,
        'sugar': selected_sugar,
        'milk': selected_milk,
        'shot': selected_shot,
    }


def format_option_summary(item):
    parts = []

    size_val = getattr(item, 'size', '') or ''
    sugar_val = getattr(item, 'sugar', '') or ''
    milk_val = getattr(item, 'milk', '') or ''
    shot_val = getattr(item, 'shot', '') or ''

    if size_val:
        parts.append(f'Size: {size_val}')
    if sugar_val:
        parts.append(f'Sugar: {sugar_val}')
    if milk_val:
        parts.append(f'Milk: {milk_val}')
    if shot_val:
        parts.append(f'Shot: {shot_val}')

    return ', '.join(parts) if parts else 'No add-ons'


def add_product_to_customer_cart(customer_id, product, size='', sugar='', milk='', shot='', quantity=1):
    cart_item = Cart.query.filter_by(
        customer_link=customer_id,
        product_link=product.id,
        size=size,
        sugar=sugar,
        milk=milk,
        shot=shot
    ).first()

    if cart_item:
        cart_item.quantity += quantity
    else:
        cart_item = Cart()
        cart_item.quantity = quantity
        cart_item.size = size
        cart_item.sugar = sugar
        cart_item.milk = milk
        cart_item.shot = shot
        cart_item.customer_link = customer_id
        cart_item.product_link = product.id
        db.session.add(cart_item)

    db.session.commit()
    return cart_item


def add_customer_usual_to_cart(customer):
    if not customer.usual_items:
        return False, f'{customer.username} has not saved a usual order yet.'

    for usual_item in customer.usual_items:
        if usual_item.product is None:
            return False, 'One of your saved usual items is no longer available.'
        if usual_item.product.in_stock < usual_item.quantity:
            return False, f'{usual_item.product.product_name} is currently out of stock.'

    for usual_item in customer.usual_items:
        add_product_to_customer_cart(
            customer.id,
            usual_item.product,
            size=usual_item.size,
            sugar=usual_item.sugar,
            milk=usual_item.milk,
            shot=usual_item.shot,
            quantity=usual_item.quantity,
        )

    return True, f'Usual items added to cart for {customer.username}.'


def create_order_and_payment(customer_id, items_list):
    """Create orders and initiate PayMongo payment.

    Args:
        customer_id: The customer's database ID.
        items_list: List of dicts with keys: product, quantity, size, sugar, milk, shot.

    Returns:
        Dict with payment_intent_id, qr_image_url, redirect_url, total.
    """
    total = 0
    for item in items_list:
        total += item['product'].current_price * item['quantity']

    total_with_shipping = total

    pi = create_payment_intent(total_with_shipping, description='Wideye Kiosk order')
    pi_id = pi['id']

    attached = attach_qrph(pi_id)

    for item in items_list:
        new_order = Order()
        new_order.quantity = item['quantity']
        new_order.size = item.get('size', '')
        new_order.sugar = item.get('sugar', '')
        new_order.milk = item.get('milk', '')
        new_order.shot = item.get('shot', '')
        new_order.price = item['product'].current_price
        new_order.status = 'Pending'
        new_order.payment_id = pi_id
        new_order.product_link = item['product'].id
        new_order.customer_link = customer_id

        db.session.add(new_order)

        product = Product.query.get(item['product'].id)
        product.in_stock -= item['quantity']

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

    return {
        'payment_intent_id': pi_id,
        'qr_image_url': qr_image_url,
        'redirect_url': redirect_url,
        'total': total_with_shipping,
    }


def create_order_cash(customer_id, items_list):
    """Create orders for a cash payment (no PayMongo call).

    Returns:
        Dict with order_group_id and total.
    """
    total = 0
    for item in items_list:
        total += item['product'].current_price * item['quantity']

    order_group_id = f'cash-{uuid.uuid4().hex[:12]}'

    for item in items_list:
        new_order = Order()
        new_order.quantity = item['quantity']
        new_order.size = item.get('size', '')
        new_order.sugar = item.get('sugar', '')
        new_order.milk = item.get('milk', '')
        new_order.shot = item.get('shot', '')
        new_order.price = item['product'].current_price
        new_order.status = 'Pending'
        new_order.payment_method = 'cash'
        new_order.payment_id = order_group_id
        new_order.product_link = item['product'].id
        new_order.customer_link = customer_id

        db.session.add(new_order)

        product = Product.query.get(item['product'].id)
        product.in_stock -= item['quantity']

    db.session.commit()

    return {
        'order_group_id': order_group_id,
        'total': total,
    }


def print_receipt(orders, total, payment_method='cash'):
    """Best-effort thermal receipt via QR204. Works for both cash and cashless orders."""
    import traceback
    try:
        from thermal_printer import QR204Printer
        with QR204Printer() as printer:
            printer.align('center')
            printer.bold()
            printer.double_size()
            printer.println('WIDEYE KIOSK')
            printer.double_size(False)
            printer.bold(False)
            subtitle = 'Cash Payment Receipt' if payment_method == 'cash' else 'Payment Receipt'
            printer.println(subtitle)
            printer.print_separator()

            printer.align('left')
            for order in orders:
                name = order.product.product_name if order.product else 'Unknown'
                printer.println(f'{name} x{order.quantity}')
                options = format_option_summary(order)
                if options and options != 'No add-ons':
                    printer.println(f'  {options}')
                printer.println(f'  PhP {order.price:.2f} each')
            printer.print_separator()

            printer.align('right')
            printer.bold()
            printer.println(f'TOTAL: PhP {total:.2f}')
            printer.bold(False)

            printer.align('center')
            printer.feed(1)
            printer.bold()
            if payment_method == 'cash':
                printer.println('PLEASE PAY AT THE COUNTER')
            else:
                printer.println('PAYMENT CONFIRMED')
            printer.bold(False)
            printer.println('Thank you for your order!')

            printer.cut()
    except Exception as exc:
        print(f'[Thermal Printer] Error (non-fatal): {exc}')
        traceback.print_exc()


def print_cash_receipt(orders, total):
    """Backward-compatible alias."""
    print_receipt(orders, total, payment_method='cash')


@views.route('/')
def home():

    items = Product.query.order_by(Product.date_added.desc()).all()

    return render_template('home.html', items=items, format_option_summary=format_option_summary,
                           parse_options=parse_options, get_product_image=get_product_image,
                           cart=Cart.query.filter_by(customer_link=current_user.id).all()
                           if current_user.is_authenticated else [])


@views.route('/add-to-cart/<int:item_id>', methods=['GET', 'POST'])
@login_required
def add_to_cart(item_id):
    item_to_add = Product.query.get(item_id)
    if item_to_add is None:
        flash('Item not found')
        return redirect(request.referrer or '/')

    selection = normalize_product_selection(
        item_to_add,
        size=request.values.get('size', ''),
        sugar=request.values.get('sugar', ''),
        milk=request.values.get('milk', ''),
        shot=request.values.get('shot', ''),
    )

    try:
        cart_item = add_product_to_customer_cart(
            current_user.id,
            item_to_add,
            size=selection['size'],
            sugar=selection['sugar'],
            milk=selection['milk'],
            shot=selection['shot']
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

    return render_template('cart.html', cart=cart, amount=amount, total=amount,
                           format_option_summary=format_option_summary,
                           get_product_image=get_product_image)


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
            'total': amount
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
            'total': amount
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
            'total': amount
        }

        return jsonify(data)


@views.route('/choose-payment')
@login_required
def choose_payment():
    customer_cart = Cart.query.filter_by(customer_link=current_user.id).all()
    if not customer_cart:
        flash('Your cart is Empty')
        return redirect('/')

    total = sum(item.product.current_price * item.quantity for item in customer_cart)

    return render_template(
        'choose_payment.html',
        cart=customer_cart,
        total=total,
        format_option_summary=format_option_summary,
        get_product_image=get_product_image,
    )


@views.route('/place-order', methods=['GET', 'POST'])
@login_required
def place_order():
    if request.method == 'GET':
        return redirect(url_for('views.choose_payment'))

    customer_cart = Cart.query.filter_by(customer_link=current_user.id).all()
    if not customer_cart:
        flash('Your cart is Empty')
        return redirect('/')

    payment_method = request.form.get('payment_method', 'cashless')

    try:
        items_list = [
            {
                'product': item.product,
                'quantity': item.quantity,
                'size': item.size,
                'sugar': item.sugar,
                'milk': item.milk,
                'shot': item.shot,
            }
            for item in customer_cart
        ]

        if payment_method == 'cash':
            result = create_order_cash(current_user.id, items_list)

            for item in customer_cart:
                db.session.delete(item)
            db.session.commit()

            orders = Order.query.filter_by(payment_id=result['order_group_id']).all()
            print_cash_receipt(orders, result['total'])

            return render_template(
                'cash_receipt.html',
                order_group_id=result['order_group_id'],
                total=result['total'],
                orders=orders,
                format_option_summary=format_option_summary,
                get_product_image=get_product_image,
            )
        else:
            result = create_order_and_payment(current_user.id, items_list)

            for item in customer_cart:
                db.session.delete(item)
            db.session.commit()

            return render_template(
                'payment_qr.html',
                payment_intent_id=result['payment_intent_id'],
                qr_image_url=result['qr_image_url'],
                redirect_url=result['redirect_url'],
                total=result['total'],
                orders=Order.query.filter_by(payment_id=result['payment_intent_id']).all(),
                format_option_summary=format_option_summary,
                get_product_image=get_product_image,
            )
    except Exception as e:
        db.session.rollback()
        print('Order error:', e)
        import traceback; traceback.print_exc()
        flash(f'Order not placed — {e}')
        return redirect('/')


@views.route('/check-payment/<payment_intent_id>')
def check_payment(payment_intent_id):
    try:
        pi = retrieve_payment_intent(payment_intent_id)
        status = pi['attributes']['status']

        if status == 'succeeded':
            updated = Order.query.filter_by(
                payment_id=payment_intent_id,
                status='Pending',
            ).update({'status': 'Accepted'})
            db.session.commit()

            # Print receipt only on the first transition (Pending → Accepted)
            if updated:
                orders = Order.query.filter_by(payment_id=payment_intent_id).all()
                total = sum(o.price * o.quantity for o in orders)
                print_receipt(orders, total, payment_method='cashless')

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
    return render_template('orders.html', orders=orders, format_option_summary=format_option_summary,
                           get_product_image=get_product_image)


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
        success_redirect=url_for('views.home'),
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

    usual_items = customer.usual_items
    if not usual_items:
        return jsonify({'ok': False, 'message': f'{customer.username} has not saved a usual order yet.'}), 400

    for usual_item in usual_items:
        if usual_item.product is None:
            return jsonify({'ok': False, 'message': 'One of the saved usual items is no longer available.'}), 400
        if usual_item.product.in_stock < usual_item.quantity:
            return jsonify({'ok': False, 'message': f'{usual_item.product.product_name} is currently out of stock.'}), 400

    payment_method = payload.get('payment_method', 'cashless')

    try:
        usual_items_data = [
            {
                'product': usual_item.product,
                'quantity': usual_item.quantity,
                'size': usual_item.size,
                'sugar': usual_item.sugar,
                'milk': usual_item.milk,
                'shot': usual_item.shot,
            }
            for usual_item in usual_items
        ]

        if payment_method == 'cash':
            result = create_order_cash(customer.id, usual_items_data)
            orders = Order.query.filter_by(payment_id=result['order_group_id']).all()
            print_cash_receipt(orders, result['total'])

            return jsonify({
                'ok': True,
                'message': f'Cash order placed for {customer.username}. Please pay at the counter.',
                'recognized_name': recognized_name,
                'redirect_url': url_for('views.usual_order_cash_receipt', order_group_id=result['order_group_id']),
            })
        else:
            payment = create_order_and_payment(customer.id, usual_items_data)

            return jsonify({
                'ok': True,
                'message': f'Order placed for {customer.username}. Proceed to payment.',
                'recognized_name': recognized_name,
                'redirect_url': url_for('views.usual_order_payment', pi_id=payment['payment_intent_id']),
            })
    except Exception as exc:
        db.session.rollback()
        print('Usual order payment error:', exc)
        return jsonify({'ok': False, 'message': 'Order could not be placed — payment initiation failed.'}), 500


@views.route('/usual-order/payment/<pi_id>')
def usual_order_payment(pi_id):
    """Payment page for usual orders — no login required."""
    orders = Order.query.filter_by(payment_id=pi_id).all()
    if not orders:
        flash('No order found for this payment.')
        return redirect('/')

    total = sum(o.price * o.quantity for o in orders)

    pi = retrieve_payment_intent(pi_id)
    next_action = pi['attributes'].get('next_action') or {}
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
        total=total,
        is_usual_order=True,
        orders=orders,
        format_option_summary=format_option_summary,
        get_product_image=get_product_image,
    )


@views.route('/usual-order/cash/<order_group_id>')
def usual_order_cash_receipt(order_group_id):
    """Cash receipt page for usual orders — no login required."""
    orders = Order.query.filter_by(payment_id=order_group_id).all()
    if not orders:
        flash('No order found.')
        return redirect('/')

    total = sum(o.price * o.quantity for o in orders)

    return render_template(
        'cash_receipt.html',
        order_group_id=order_group_id,
        total=total,
        orders=orders,
        is_usual_order=True,
        format_option_summary=format_option_summary,
        get_product_image=get_product_image,
    )


@views.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_query = request.form.get('search')
        items = Product.query.filter(Product.product_name.ilike(f'%{search_query}%')).all()
        return render_template('search.html', items=items, cart=Cart.query.filter_by(customer_link=current_user.id).all()
                           if current_user.is_authenticated else [],
                           get_product_image=get_product_image)

    return render_template('search.html')















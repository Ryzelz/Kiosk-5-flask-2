from flask import Blueprint, render_template, flash, redirect, request, url_for, send_from_directory, jsonify
from .forms import LoginForm, SignUpForm, PasswordChangeForm, UsualOrderForm
from .models import Customer, Product, UsualOrderItem
from . import db
from flask_login import login_user, login_required, logout_user, current_user
from .face_profiles import get_face_profile_dir, list_saved_face_images, normalize_face_profile_name
from .views import get_product_image, parse_options, format_option_summary, normalize_product_selection


auth = Blueprint('auth', __name__)


def sync_primary_usual_product(customer):
    first_item = customer.usual_items[0] if customer.usual_items else None
    customer.usual_product_id = first_item.product_link if first_item else None


@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    form = SignUpForm()
    if form.validate_on_submit():
        email = form.email.data
        username = form.username.data
        password1 = form.password1.data
        password2 = form.password2.data

        if password1 == password2:
            new_customer = Customer()
            new_customer.email = email
            new_customer.username = username
            new_customer.password = password2
            new_customer.is_admin = Customer.query.filter_by(is_admin=True).first() is None
            new_customer.face_profile_name = normalize_face_profile_name(username)

            try:
                db.session.add(new_customer)
                db.session.commit()
                flash('Account Created Successfully, You can now Login')
                return redirect('/login')
            except Exception as e:
                db.session.rollback()
                print(e)

                existing = Customer.query.filter_by(email=email).first()
                if existing:
                    flash('Account Not Created!! Email already exists')
                else:
                    flash(f'Account Not Created!! {e}')

            form.email.data = ''
            form.username.data = ''
            form.password1.data = ''
            form.password2.data = ''

    return render_template('signup.html', form=form)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        customer = Customer.query.filter_by(email=email).first()

        if customer:
            if customer.verify_password(password=password):
                login_user(customer)
                return redirect('/')
            else:
                flash('Incorrect Email or Password')

        else:
            flash('Account does not exist please Sign Up')

    return render_template('login.html', form=form)


@auth.route('/logout', methods=['GET', 'POST'])
@login_required
def log_out():
    logout_user()
    return redirect('/')


@auth.route('/profile/<int:customer_id>', methods=['GET', 'POST'])
@login_required
def profile(customer_id):
    if current_user.id != customer_id:
        flash('You can only edit your own profile.')
        return redirect(url_for('auth.profile', customer_id=current_user.id))

    customer = Customer.query.get_or_404(customer_id)

    if not customer.face_profile_name:
        customer.face_profile_name = normalize_face_profile_name(customer.username)
        db.session.commit()

    form = UsualOrderForm()
    products = Product.query.order_by(Product.product_name.asc()).all()
    face_images = list_saved_face_images(customer.face_profile_name)
    saved_usual_items = customer.usual_items

    if request.method == 'POST' and request.form.get('remove_usual_item_id'):
        usual_item = UsualOrderItem.query.filter_by(
            id=request.form.get('remove_usual_item_id'),
            customer_link=customer.id,
        ).first_or_404()
        product_name = usual_item.product.product_name if usual_item.product else 'Saved item'
        db.session.delete(usual_item)
        db.session.flush()
        sync_primary_usual_product(customer)
        db.session.commit()
        flash(f'{product_name} removed from your usual order.')
        return redirect(url_for('auth.profile', customer_id=customer.id))

    if request.method == 'POST' and 'remove_usual' in request.form:
        for usual_item in list(customer.usual_items):
            db.session.delete(usual_item)
        customer.usual_product_id = None
        db.session.commit()
        flash('Your usual order has been removed.')
        return redirect(url_for('auth.profile', customer_id=customer.id))

    if request.method == 'POST' and request.form.get('add_usual_product_id'):
        selected_product = Product.query.get_or_404(request.form.get('add_usual_product_id'))
        selection = normalize_product_selection(
            selected_product,
            size=request.form.get('size', ''),
            sugar=request.form.get('sugar', ''),
            milk=request.form.get('milk', ''),
            shot=request.form.get('shot', ''),
        )

        existing_item = UsualOrderItem.query.filter_by(
            customer_link=customer.id,
            product_link=selected_product.id,
            size=selection['size'],
            sugar=selection['sugar'],
            milk=selection['milk'],
            shot=selection['shot'],
        ).first()

        if existing_item:
            existing_item.quantity += 1
        else:
            db.session.add(UsualOrderItem(
                customer_link=customer.id,
                product_link=selected_product.id,
                quantity=1,
                size=selection['size'],
                sugar=selection['sugar'],
                milk=selection['milk'],
                shot=selection['shot'],
            ))

        if not customer.face_profile_name:
            customer.face_profile_name = normalize_face_profile_name(customer.username)

        db.session.flush()
        sync_primary_usual_product(customer)
        db.session.commit()
        flash(f'{selected_product.product_name} added to your usual order.')
        return redirect(url_for('auth.profile', customer_id=customer.id))

    return render_template(
        'profile.html',
        customer=customer,
        form=form,
        face_images=face_images,
        products=products,
        saved_usual_items=saved_usual_items,
        get_product_image=get_product_image,
        parse_options=parse_options,
        format_option_summary=format_option_summary,
    )


@auth.route('/profile/<int:customer_id>/train-usual-face')
@login_required
def train_usual_face(customer_id):
    if current_user.id != customer_id:
        flash('You can only train your own profile.')
        return redirect(url_for('auth.profile', customer_id=current_user.id))

    customer = Customer.query.get_or_404(customer_id)
    customer.face_profile_name = normalize_face_profile_name(customer.face_profile_name or customer.username)
    db.session.commit()

    return render_template(
        'face_camera.html',
        page_title='Train usual face',
        mode='training',
        back_url=url_for('auth.profile', customer_id=customer.id),
        training_customer_id=customer.id,
        preview_url=url_for('views.preview_face_frame'),
        training_steps=[
            {'key': 'front', 'label': 'Look straight', 'required': 5},
            {'key': 'left', 'label': 'Face left', 'required': 4},
            {'key': 'right', 'label': 'Face right', 'required': 4},
            {'key': 'blink', 'label': 'Blink', 'required': 3},
        ],
        capture_url=url_for('auth.capture_training_face_frame', customer_id=customer.id),
        reset_url=url_for('auth.reset_training_face_capture', customer_id=customer.id),
        success_redirect=url_for('auth.profile', customer_id=customer.id),
    )


@auth.route('/profile/<int:customer_id>/train-usual-face/reset', methods=['POST'])
@login_required
def reset_training_face_capture(customer_id):
    if current_user.id != customer_id:
        return jsonify({'ok': False, 'message': 'You can only train your own profile.'}), 403

    customer = Customer.query.get_or_404(customer_id)

    try:
        from yolov10 import reset_training_capture
        reset_training_capture(customer.face_profile_name or customer.username)
    except Exception as exc:
        print(exc)
        return jsonify({'ok': False, 'message': 'Could not reset the face training session.'}), 500

    return jsonify({'ok': True, 'message': 'Face training session reset.'})


@auth.route('/profile/<int:customer_id>/train-usual-face/capture', methods=['POST'])
@login_required
def capture_training_face_frame(customer_id):
    if current_user.id != customer_id:
        return jsonify({'ok': False, 'message': 'You can only train your own profile.'}), 403

    customer = Customer.query.get_or_404(customer_id)
    payload = request.get_json(silent=True) or {}
    frame_data = payload.get('frame')
    stage_name = payload.get('stage')

    try:
        from yolov10 import capture_training_frame
        result = capture_training_frame(customer.face_profile_name or customer.username, stage_name, frame_data)
    except Exception as exc:
        print(exc)
        return jsonify({'ok': False, 'message': 'Face training could not be completed.'}), 500

    status_code = 200 if result.get('ok') else 400
    return jsonify(result), status_code


@auth.route('/profile/<int:customer_id>/face-image/<path:filename>')
@login_required
def profile_face_image(customer_id, filename):
    if current_user.id != customer_id:
        flash('You can only view your own saved face samples.')
        return redirect(url_for('auth.profile', customer_id=current_user.id))

    customer = Customer.query.get_or_404(customer_id)
    face_dir = get_face_profile_dir(customer.face_profile_name or customer.username)
    return send_from_directory(str(face_dir), filename)


@auth.route('/change-password/<int:customer_id>', methods=['GET', 'POST'])
@login_required
def change_password(customer_id):
    form = PasswordChangeForm()
    customer = Customer.query.get(customer_id)
    if form.validate_on_submit():
        current_password = form.current_password.data
        new_password = form.new_password.data
        confirm_new_password = form.confirm_new_password.data

        if customer.verify_password(current_password):
            if new_password == confirm_new_password:
                customer.password = confirm_new_password
                db.session.commit()
                flash('Password Updated Successfully')
                return redirect(f'/profile/{customer.id}')
            else:
                flash('New Passwords do not match!!')

        else:
            flash('Current Password is Incorrect')

    return render_template('change_password.html', form=form)








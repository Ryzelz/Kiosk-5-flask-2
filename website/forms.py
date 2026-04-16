from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, PasswordField, EmailField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired, InputRequired, Optional, length, NumberRange
from flask_wtf.file import FileField, FileRequired, MultipleFileField


class SignUpForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired()])
    username = StringField('Username', validators=[DataRequired(), length(min=2)])
    password1 = PasswordField('Enter Your Password', validators=[DataRequired(), length(min=6)])
    password2 = PasswordField('Confirm Your Password', validators=[DataRequired(), length(min=6)])
    submit = SubmitField('Sign Up')


class LoginForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired()])
    password = PasswordField('Enter Your Password', validators=[DataRequired()])
    submit = SubmitField('Log in')


class PasswordChangeForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired(), length(min=6)])
    new_password = PasswordField('New Password', validators=[DataRequired(), length(min=6)])
    confirm_new_password = PasswordField('Confirm New Password', validators=[DataRequired(), length(min=6)])
    change_password = SubmitField('Change Password')


class ProfileUpdateForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired()])
    username = StringField('Username', validators=[DataRequired(), length(min=2)])
    update_account = SubmitField('Save Account Changes')


class DeleteAccountForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired(), length(min=6)])
    delete_account = SubmitField('Delete Account')


class UsualOrderForm(FlaskForm):
    save_usual = SubmitField('Add to usual')
    remove_usual = SubmitField('Remove all usual')


class ShopItemsForm(FlaskForm):
    product_name = StringField('Name of Product', validators=[DataRequired()])
    current_price = FloatField('Current Price', validators=[DataRequired()])
    previous_price = FloatField('Previous Price', validators=[DataRequired()])
    in_stock = IntegerField('In Stock', validators=[InputRequired(), NumberRange(min=0)])
    size = StringField('Size Options (comma-separated)', validators=[Optional()])
    sugar = StringField('Sugar Options (comma-separated)', validators=[Optional()])
    milk = StringField('Milk Options (comma-separated)', validators=[Optional()])
    shot = StringField('Shot Options (comma-separated)', validators=[Optional()])
    product_picture = MultipleFileField('Product Pictures')
    flash_sale = BooleanField('Flash Sale')

    add_product = SubmitField('Add Product')
    update_product = SubmitField('Update')


class UpdateShopItemsForm(FlaskForm):
    product_name = StringField('Name of Product', validators=[Optional()])
    current_price = FloatField('Current Price', validators=[Optional(), NumberRange(min=0)])
    previous_price = FloatField('Previous Price', validators=[Optional(), NumberRange(min=0)])
    in_stock = IntegerField('In Stock', validators=[Optional(), NumberRange(min=0)])
    size = StringField('Size Options (comma-separated)', validators=[Optional()])
    sugar = StringField('Sugar Options (comma-separated)', validators=[Optional()])
    milk = StringField('Milk Options (comma-separated)', validators=[Optional()])
    shot = StringField('Shot Options (comma-separated)', validators=[Optional()])
    product_picture = MultipleFileField('Product Pictures')
    flash_sale = BooleanField('Flash Sale')

    update_product = SubmitField('Save Changes')


class AdminCustomerUpdateForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired()])
    username = StringField('Username', validators=[DataRequired(), length(min=2)])
    is_admin = BooleanField('Admin Access')
    update_customer = SubmitField('Save Changes')


class OrderForm(FlaskForm):
    order_status = SelectField('Order Status', choices=[('Pending', 'Pending'), ('Accepted', 'Accepted'),
                                                        ('Out for delivery', 'Out for delivery'),
                                                        ('Delivered', 'Delivered'), ('Canceled', 'Canceled')])

    update = SubmitField('Update Status')






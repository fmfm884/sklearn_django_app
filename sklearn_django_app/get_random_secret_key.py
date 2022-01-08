from django.core.management.utils import get_random_secret_key

secret_key = get_random_secret_key()
text = 'SECRET_KEY = \'{0}\''.format(secret_key)
try:
    with open('./sklearn_django_app/local_settings.py', 'x') as f:
        print(text, file=f)
except FileExistsError:
    print('[Info]Secret_key will not be updated because the \'local_settings.py\' already exists.')
    pass
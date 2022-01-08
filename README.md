# sklearn_django_app
sklearnのアルゴリズムを検証するアプリ

### Environment
```
conda env create -n 新たな環境名 -f env.yml
```

### Prepare Django_Secret_Key
```
python sklearn_django_app/get_random_secret_key.py
```

### Run Web Server 
```
python manage.py runserver
```

[こちら](http://localhost:8000/home/)にアクセスすると利用できます。
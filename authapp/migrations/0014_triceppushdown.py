# Generated by Django 5.0.4 on 2025-01-22 13:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authapp', '0013_tricepdip'),
    ]

    operations = [
        migrations.CreateModel(
            name='TricepPushdown',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('category', models.CharField(max_length=50)),
            ],
        ),
    ]

# Generated by Django 5.0.4 on 2025-02-09 11:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authapp', '0019_benchpress'),
    ]

    operations = [
        migrations.CreateModel(
            name='Hammercurl',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('category', models.CharField(max_length=50)),
            ],
        ),
    ]

from django.contrib import admin
from .models import *
# Register your models here.

admin.site.register(Game)
admin.site.register(Player)
admin.site.register(Card_Player)
admin.site.register(Card_Community)
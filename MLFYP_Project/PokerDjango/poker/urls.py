from django.contrib import admin
from django.urls import include, path
from . import views
# from main_files.holdem.DQN import *
from .views import table_view, GameViewSet, PlayerViewSet, CommunityCardViewSet, PlayerCardViewSet
from rest_framework import routers
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.authtoken.views import obtain_auth_token
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'api/games', GameViewSet, basename='game')
router.register(r'api/players', PlayerViewSet, basename='player')
router.register(r'api/community_cards', CommunityCardViewSet, basename='community_cards')
router.register(r'api/player_cards', PlayerCardViewSet, basename='player_cards')

urlpatterns = [
    # path('', table_view.as_view(), name="table_view"),
    # path('api/', include(router.urls)),
    # path('react/', TemplateView.as_view(template_name='poker/react.html')),

    path(r'', include(router.urls)),
    path(r'api/', include('rest_framework.urls', namespace='rest_framework')),
    path('api-token-auth/', obtain_auth_token, name='api-token-auth')
    # path('api/player_cards/', PlayerCardView, name='player_cards')

] + static(settings.CARDS_URL, document_root=settings.CARDS_ROOT)
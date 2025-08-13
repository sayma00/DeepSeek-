from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    # Chat UI page
    path('', views.chat_ui, name='chat_ui'),

    # API endpoint to send messages/files to the model
    path('chat/', views.send_to_ollama, name='send_to_ollama'),
]

# ✅ Serve media files (only in DEBUG mode)
# Needed so JSON download links work in the browser during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from apps.accounts.models import Profile

class Command(BaseCommand):
    help = "Ensure that every user has a Profile."

    def handle(self, *args, **kwargs):
        created_count = 0
        for user in User.objects.all():
            profile, created = Profile.objects.get_or_create(user=user)
            if created:
                created_count += 1
                self.stdout.write(self.style.SUCCESS(f"✅ Created profile for {user.username}"))
        if created_count == 0:
            self.stdout.write(self.style.WARNING("ℹ️ All users already have profiles."))
        else:
            self.stdout.write(self.style.SUCCESS(f"🎉 {created_count} profiles created."))
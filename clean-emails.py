#!/usr/bin/env python3
import sys
from subprocess import run

# Map of emails to replace
EMAIL_MAPPING = {
    "davidberf@gmail.com": "anonymous@example.com",
    "david.erf@berkeley.edu": "anonymous@university.edu",
    "13736190+david-erf@users.noreply.github.com": "anonymous@users.noreply.github.com"
}

def clean_message(message):
    for old_email, new_email in EMAIL_MAPPING.items():
        message = message.replace(old_email, new_email)
    return message

if __name__ == "__main__":
    message = sys.stdin.read()
    print(clean_message(message))

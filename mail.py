# IMPORT

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT
load_dotenv()
# =========================================================
# GMAIL CONFIGURATION
GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# ==================# SEND EMAIL=======================================

def send_email(receiver_email, subject, body):
    try:

        # CHECK GMAIL CONFIGURATION

        if not GMAIL_EMAIL:
            print("ERROR: GMAIL_EMAIL is missing.")
            return False

        if not GMAIL_APP_PASSWORD:
            print("ERROR: GMAIL_APP_PASSWORD is missing.")
            return False


        # -------------------------------------------------
        # EMAIL INFORMATION
        # -------------------------------------------------

        print("\n" + "=" * 60)
        print("EMAIL SENDING")
        print("=" * 60)
        print("Sender:", GMAIL_EMAIL)
        print("Receiver:", receiver_email)


        # -------------------------------------------------
        # CREATE EMAIL
        # -------------------------------------------------

        message = MIMEMultipart()
        message["From"] = GMAIL_EMAIL
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))


        # -------------------------------------------------
        # CONNECT TO GMAIL
        # -------------------------------------------------

        print("Connecting to Gmail...")

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.ehlo()

      # -------------------------------------------------
        # LOGIN
       

        print("Logging into Gmail...")
        server.login(GMAIL_EMAIL, GMAIL_APP_PASSWORD)
    # -------------------------------------------------
        # SEND EMAIL

        print("Sending email...")

        server.sendmail(
            GMAIL_EMAIL,
            receiver_email,
            message.as_string()
        )
        # -------------------------------------------------
        # CLOSE CONNECTION
        # -------------------------------------------------

        server.quit()

        print("Email sent successfully.")
        print("=" * 60)
        return True
   # -----------------------------------------------------
    # GMAIL AUTHENTICATION ERROR
    # -----------------------------------------------------

    except smtplib.SMTPAuthenticationError as e:
        print("\nGMAIL AUTHENTICATION ERROR")
        print(e)
        return False
   # -----------------------------------------------------
    # SMTP ERROR

    except smtplib.SMTPException as e:
        print("\nSMTP ERROR")
        print(e)
        return False

    # GENERAL ERROR
    # -----------------------------------------------------

    except Exception as e:
        print("\nEMAIL ERROR")
        print(e)
        return False
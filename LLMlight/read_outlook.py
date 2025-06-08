import os
import csv
import pypff
from datetime import datetime

#%%

import os
import pypff
from email.message import EmailMessage
from datetime import datetime

def extract_eml_from_pst(pst_path, output_dir="exported_eml"):
    os.makedirs(output_dir, exist_ok=True)

    pst = pypff.file()
    pst.open(pst_path)

    root = pst.get_root_folder()
    email_count = 0

    def walk_folder(folder, folder_path):
        nonlocal email_count
        folder_name = folder.name or "root"
        current_path = os.path.join(folder_path, folder_name)
        os.makedirs(current_path, exist_ok=True)

        # Extract messages
        for i in range(folder.number_of_messages):
            message = folder.get_message(i)
            msg = EmailMessage()

            msg['Subject'] = message.subject or ''
            msg['From'] = message.sender_name or ''
            msg['To'] = message.display_to or ''
            msg['CC'] = message.display_cc or ''
            msg['BCC'] = message.display_bcc or ''
            msg['Date'] = message.delivery_time.isoformat() if message.delivery_time else ''

            body = message.plain_text_body or message.html_body or ''
            msg.set_content(body)

            # Save as .eml
            filename = f"email_{email_count:05d}.eml"
            filepath = os.path.join(current_path, filename)
            with open(filepath, "wb") as f:
                f.write(bytes(msg))

            email_count += 1

        # Recurse into subfolders
        for i in range(folder.number_of_sub_folders):
            subfolder = folder.get_sub_folder(i)
            walk_folder(subfolder, current_path)

    walk_folder(root, output_dir)
    pst.close()
    print(f"Done: Extracted {email_count} emails to '{output_dir}'")

    return email_count



#%%
from libratom.lib.pff import PffArchive
import os
import json
import re
from tqdm import tqdm
from unidecode import unidecode
import logging
from collections import defaultdict

"""
    Needs a .json file email_list.json with the following format:
    blacklist - list of blacklisted sender emails
    whitelist - list of whitelisted sender emails
"""

logging.info("Logging is configured.")

LIMIT = 100000

# Load whitelist and blacklist from matt_list.yml
with open("email_list.json", "r") as file:
    lists = json.load(file)
    WHITELIST = lists.get("whitelist", [])
    BLACKLIST = lists.get("blacklist", [])

def is_good_email(message, sender_email):
    if sender_email in WHITELIST:
        return True
    if sender_email in BLACKLIST:
        return False
    return True

def extract_header_info(headers):
    # Extract the sender name and email address from the headers
    sender_re = re.search(r"From: (.+?) <(.+?)>", headers)
    if sender_re:
        sender_name = sender_re.group(1).strip('"')
        sender_email = sender_re.group(2)
    else:
        sender_name = "Unknown Sender"
        sender_email = "Unknown Email"

    # Extract the timestamp from the headers
    timestamp_re = re.search(r"Date: (.+)", headers)
    if timestamp_re:
        timestamp = timestamp_re.group(1)
    else:
        timestamp = "Unknown Timestamp"

    return sender_name, sender_email, timestamp

def clean_subject(subject):
    subject_ascii = unidecode(subject)
    clean_subject = re.sub(r'[\\/*?:"<>|]', "_", subject_ascii).strip().rstrip(". ")
    return clean_subject

def save_body(message, message_folder):
    # Extract the email body (plain text, HTML, or RTF)
    if message.plain_text_body:
        body = message.plain_text_body
        body_file = os.path.join(message_folder, "body.txt")
        with open(body_file, "w", encoding="utf-8") as f:
            f.write(body)
    elif message.html_body:
        body = message.html_body
        body_file = os.path.join(message_folder, "body.html")
        with open(body_file, "w", encoding="utf-8") as f:
            f.write(body)
    elif message.rtf_body:
        try:
            body = message.rtf_body
            body_file = os.path.join(message_folder, "body.rtf")
            # Decode RTF body from bytes to string
            with open(body_file, "wb") as msg_file:
                msg_file.write(body)
        except UnicodeEncodeError:
            # Handle encoding error by using a different encoding
            logging.error("Encoding error encountered while processing RTF body.")
            body = "Encoding error: Unable to extract body content."
            body_file = os.path.join(message_folder, "body.txt")
            with open(body_file, "w", encoding="utf-8") as f:
                f.write(body)
    else:
        logging.warning("No body content found")
        body = "No body content available"
        body_file = os.path.join(message_folder, "body.txt")
        with open(body_file, "w", encoding="utf-8") as f:
            f.write(body)

    return body_file

def save_attachments(message, message_folder):
    # Initialize a list to store attachment paths
    attachment_paths = []

    try:
        # Check if the message has attachments
        if message.attachments:
            for attachment in message.attachments:
                attachment_name = attachment.name or "Unnamed_attachment"
                attachment_path = os.path.join(message_folder, attachment_name)
                with open(attachment_path, "wb") as f:
                    f.write(attachment.read_buffer(attachment.get_size()))
                attachment_paths.append(attachment_path)
    except OSError as e:
        logging.error("Error saving attachment %s %s: %s", message.subject, attachment_name, e)

    return attachment_paths

def download_emails(pst_file_path, output_folder):
    """Extract and save the first 10 email bodies from the given .pst file."""
    # Open the .pst file using PffArchive from libratom
    with PffArchive(pst_file_path) as archive:
        # Initialize a counter to keep track of the number of processed emails
        email_count = 0
        name_counts = defaultdict(int)
        senders = set()
        email_list = []

        # Iterate through all folders in the .pst file
        for folder in archive.folders():
            if folder.name != "Inbox":
                continue
            # Loop through each message in the folder
            for index in tqdm(range(folder.get_number_of_sub_messages())):
                # Get the message using the index
                message = folder.get_sub_message(index)

                if email_count >= LIMIT:
                    break

                if message.subject and message.subject == "Your daily briefing":
                    continue # spooky stuff

                if not message.transport_headers:
                    logging.warning("No headers found for message %s", message.subject)
                    continue

                header_str = message.transport_headers.strip()
                sender_name, sender_email, timestamp = extract_header_info(header_str)

                # skip bad emails
                if not is_good_email(message, sender_email):
                    continue

                subject = message.subject or "(No Subject)"
                clean_subject_name = clean_subject(subject)

                # Check for duplicate subject names and append a number to the name
                if clean_subject_name in name_counts:
                    name_counts[clean_subject_name] += 1
                    clean_subject_name = f"{clean_subject_name}_{name_counts[clean_subject_name]}"
                else:
                    name_counts[clean_subject_name] = 1

                message_folder = os.path.join(output_folder, folder.name, clean_subject_name)
                try:
                    os.makedirs(message_folder, exist_ok=True)
                except OSError as e:
                    logging.error("Error creating folder %s: subject %s clean %s", message_folder, subject, clean_subject_name)
                    continue

                body_file = save_body(message, message_folder)
                attachment_paths = save_attachments(message, message_folder)

                # Add attachment paths to the email dictionary
                senders.add(sender_email)
                email_list.append({
                    "subject": subject,
                    "sender_name": sender_name,
                    "sender_email": sender_email,
                    "body": body_file,
                    "timestamp": timestamp,
                    "attachments": attachment_paths
                })

                email_count += 1

    print("SENDERS", len(senders))
    print("POST FILTER EMAIL COUNT", len(email_list))

    with open("emails.json", "w", encoding="utf-8") as json_file:
        json.dump(email_list, json_file, indent=4)

def clean_workspace(output_folder):
    if os.path.exists(output_folder):
        for root, dirs, files in os.walk(output_folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.makedirs(output_folder, exist_ok=True)

def main():
    # Replace with your .pst file path
    pst_file_path = 'backup.pst'
    output_folder = "./email_data"
    clean_workspace(output_folder)

    download_emails(pst_file_path, output_folder)
    
#%%
def process_pst_file(pst_path, output_csv='emails.csv', attachments_dir='attachments'):
    # Ensure attachments directory exists
    os.makedirs(attachments_dir, exist_ok=True)

    pst_file = pypff.file()
    pst_file.open(pst_path)

    root_folder = pst_file.get_root_folder()
    emails = []

    def sanitize_folder_name(name):
        return "".join(c if c.isalnum() or c in " _-" else "_" for c in name)

    def parse_folder(folder, folder_path):
        current_path = os.path.join(folder_path, sanitize_folder_name(folder.name or "Root"))
        os.makedirs(current_path, exist_ok=True)

        for i in range(folder.get_number_of_sub_messages()):
            message = folder.get_sub_message(i)
            delivery_time = message.delivery_time
            timestamp = delivery_time.strftime('%Y-%m-%d %H:%M:%S') if delivery_time else ''

            email = {
                'subject': message.subject,
                'sender': message.sender_name,
                'sender_email': message.sender_email_address,
                'to': message.display_to,
                'cc': message.display_cc,
                'bcc': message.display_bcc,
                'body': message.plain_text_body,
                'delivery_time': timestamp,
                'attachments': []
            }

            # Handle attachments
            for j in range(message.number_of_attachments):
                attachment = message.get_attachment(j)
                filename = attachment.get_long_filename() or attachment.get_filename()
                if not filename:
                    filename = f"attachment_{j}.bin"
                filename = sanitize_folder_name(filename)
                filepath = os.path.join(current_path, filename)

                with open(filepath, 'wb') as f:
                    f.write(attachment.read_buffer())

                email['attachments'].append(filepath)

            emails.append(email)

        # Recurse into subfolders
        for j in range(folder.get_number_of_sub_folders()):
            parse_folder(folder.get_sub_folder(j), current_path)

    # Start parsing from root
    parse_folder(root_folder, attachments_dir)

    # Export to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['subject', 'sender', 'sender_email', 'to', 'cc', 'bcc', 'delivery_time', 'body', 'attachments']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for email in emails:
            email['attachments'] = "; ".join(email['attachments'])
            writer.writerow(email)

    pst_file.close()
    print(f"Export complete: {len(emails)} emails to {output_csv}, attachments in '{attachments_dir}'.")

    return emails

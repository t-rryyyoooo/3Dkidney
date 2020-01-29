import os.path
import datetime
import smtplib
import mimetypes
import argparse
from argparse import ArgumentParser
from email.message import EmailMessage
from email.policy import SMTP

args = None


def ParseArgs():
    parser = argparse.ArgumentParser()
    #parser.add_argument("from_addr", help="from address")
    #parser.add_argument("to_addr", help="to address")
    parser.add_argument("file_path", help="C:/s/...")
    
    args = parser.parse_args()
    
    return args

args = ParseArgs()

def create_message(from_addr,to_addr,subject,body,mine,attach_file):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)
    
    file = open((attach_file['path']),'rb')
    file_read = file.read()
    msg.add_attachment(file_read,maintype=mine["type"],subtype=mine["subtype"],filename=attach_file["name"])
    file.close()
    
    return msg

def sendGmail(from_addr,to_addr,msg):
    smtp = smtplib.SMTP_SSL("smtp.gmail.com",465)
    smtp.ehlo()
    smtp.login("tobdas3d@gmail.com","wanwan01201216")
    smtp.sendmail(from_addr,to_addr,msg.as_string())
    smtp.quit()

    
from_addr = "tobdas3d@gmail.com"
to_addr = "rryyyoooo@i.softbank.jp"
subject = "result"
body = "result\n"
mine={"type":"text","subtype":"comma-separated-values"}
attach_file={"name":os.path.basename(args.file_path),"path":(args.file_path)}
msg = create_message(from_addr,to_addr,subject,body,mine,attach_file)
sendGmail(from_addr,to_addr,msg)
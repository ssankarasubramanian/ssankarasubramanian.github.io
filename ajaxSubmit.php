<?php


$name =$_POST ['name'];
$email =$_POST ['email'];
$message =$_POST ['message'];

$header = "From:" . $email . "\r\n";
$header .= "X-Mailer: PHP /" . phpverison() . "\r\n";
$header .= "Mime Version: 1.0 \r\n" ;
$header .= "Content Type: text/plain";

$comment ="This message has been sent by " . $name . "\r\n";
$comment ="E-mail is:" . $email . "\r\n";
$comment ="The message is:" . $message . "\r\n";

$for = "ssankar1991@outlook.com";
$subject = "Contact from website";

mail($subject , utf8_decode($comment), $header);

echo json_encode(array(
    'Message' => sprintf("your message has been sent %s", $name);
  ));


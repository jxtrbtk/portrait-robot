<?php header('Access-Control-Allow-Origin: *'); ?>
<?php
$id   = $_GET["id"];
//$data = $_POST["data"];
//$name = $_POST["name"];



?>
<?=$_SERVER['REQUEST_METHOD'] ?>
<?php
    phpinfo(INFO_VARIABLES);
?>OK
<? print_r($_POST) ?>
<?= $id ?>
<br/>

<?php
echo md5(uniqid("",true));
?>        
<?php
print("[");
$path = "data/in";
$dir = opendir($path);
$idx=0;
$separator="";
while($file = readdir($dir))
{
    if ($file != '.' and $file != '..' and (strpos($file, '.txt') !== false) and (strpos($file, 'TASK_') !== false))
    {
        if(file_exists ($path ."/". $file))
        {
            print($separator."\"".str_replace("TASK_","",explode(".",$file)[0])."\"");
            $idx++;
            $separator=",";
        }
    }
}
closedir($dir);
print("]");
?>
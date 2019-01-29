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
	    $tab_filename = explode(".",$file);
	    $base_name = $tab_filename[0];
	    $task_id = str_replace("TASK_","",$base_name);
	    print($separator."\"".$task_id."\"");
            $idx++;
            $separator=",";
        }
    }
}
closedir($dir);
print("]");
?>
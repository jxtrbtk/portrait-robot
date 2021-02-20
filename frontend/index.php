<!DOCTYPE html>
<html lang="en">
    <head>
        <title>Portrait-Robot</title>
        <meta name="description" content="This person does not exist" />
        <meta charset="utf-8">
        <meta name="viewport"  content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>

        <script>
         "use strict";
         var taskFlag = 0;
         var taskNo = 0;
         var taskId = '';
         var taskMessage = 'start...';
         var codeDefault = '....-....-..';
         var projectCode = '';
         var projectStep = '';
         var projectScore = '';
         var projectSide = '';
                  
         function setTaskFlag(val)
         {
            taskFlag=parseInt(val);
            var msgBox = document.getElementById('message');
            var scoreBox = document.getElementById('score');
            var statusBox = document.getElementById('dacode');
            var statusBtn = document.getElementById('s0');
            var mainZone = document.getElementById('main');
            var target = document.getElementById('target');
            var badge1 = document.getElementById('badge1');    			 
            var badge2 = document.getElementById('badge2');    			 
            var badge3 = document.getElementById('badge3');    			 
            var s1Btn =  document.getElementById('s1'); 
            var s2Btn =  document.getElementById('s2'); 
            var s3Btn =  document.getElementById('s3'); 

            if(taskFlag==0)
            { 
                statusBtn.disabled = false; 
                statusBox.style.backgroundColor = 'rgb(255, 255, 255)';
                statusBox.readOnly = false;
                mainZone.style.visibility = 'hidden';
                s1Btn.style.visibility = 'hidden';
                s2Btn.style.visibility = 'hidden';
                s3Btn.style.visibility = 'hidden';
                badge1.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge2.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge3.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
            }
            if(taskFlag==1)
            { 
                statusBtn.disabled = true; 
                statusBox.style.backgroundColor = 'rgb(222, 222, 222)';
                statusBox.readOnly = true;
                mainZone.style.visibility = 'hidden';
                s1Btn.style.visibility = 'hidden';
                s2Btn.style.visibility = 'hidden';
                s3Btn.style.visibility = 'hidden';
                badge1.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge2.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge3.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
            }
            if(taskFlag==2)
            { 
                statusBtn.disabled = true; 
                statusBox.style.backgroundColor = 'rgb(156, 218, 252)';
                statusBox.readOnly = true;
                mainZone.style.visibility = 'hidden';
                s1Btn.style.visibility = 'hidden';
                s2Btn.style.visibility = 'hidden';
                s3Btn.style.visibility = 'hidden';
                badge1.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge2.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge3.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
            }
            if(taskFlag==3)
            { 
                statusBtn.disabled = true; 
                statusBox.readOnly = true;
			    if (statusBox.style.backgroundColor.toString() == 'rgb(255, 148, 48)')
    			{ 
            		statusBox.style.backgroundColor = 'rgb(255, 110, 48)';
			    } else { 
                    statusBox.style.backgroundColor = 'rgb(255, 148, 48)';
                }                
                mainZone.style.visibility = 'hidden';
                s1Btn.style.visibility = 'hidden';
                s2Btn.style.visibility = 'hidden';
                s3Btn.style.visibility = 'hidden';
                badge1.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge2.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge3.setAttribute('style','opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
            }
            if(taskFlag==4)
            { 
                statusBtn.disabled = true; 
                statusBox.read1Only = true;
			    if (statusBox.style.backgroundColor.toString() == 'rgb(249, 255, 102)')
    			{ 
            		statusBox.style.backgroundColor = 'rgb(232, 239, 33)';
			    } else { 
                    statusBox.style.backgroundColor = 'rgb(249, 255, 102)';
                }                
                mainZone.style.visibility = 'hidden';
                s1Btn.style.visibility = 'hidden';
                s2Btn.style.visibility = 'hidden';
                s3Btn.style.visibility = 'hidden';
                badge1.setAttribute('style','padding:5px; opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge2.setAttribute('style','padding:5px; opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge3.setAttribute('style','padding:5px; opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
            }
            if(taskFlag==5)
            { 
                statusBtn.disabled = true; 
                statusBox.style.backgroundColor = 'rgb(132, 255, 138)';
                statusBox.readOnly = true;
                mainZone.visibility = 'hidden';
                s1Btn.style.visibility = 'hidden';
                s2Btn.style.visibility = 'hidden';
                s3Btn.style.visibility = 'hidden';
                badge1.setAttribute('style','padding:5px; opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge2.setAttribute('style','padding:5px; opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge3.setAttribute('style','padding:5px; opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
            }
            if(taskFlag==6)
            { 
                target.src='data/work/IMG_0_'+projectCode+'_'+projectStep+'.png';
                badge1.src='data/work/IMG_1_'+projectCode+'_'+projectStep+'.png';
                badge2.src='data/work/IMG_2_'+projectCode+'_'+projectStep+'.png';
                badge3.src='data/work/IMG_3_'+projectCode+'_'+projectStep+'.png';
		if(projectScore!='')
		{ 
			scoreBox.innerHTML =  projectScore + '%'; 
		}
                statusBtn.disabled = true; 
                statusBox.value = projectCode; 
                statusBox.style.backgroundColor = 'rgb(132, 255, 138)';
                statusBox.readOnly = true;
                mainZone.style.visibility = 'visible';
                s1Btn.style.visibility = 'visible';
                s2Btn.style.visibility = 'visible';
                s3Btn.style.visibility = 'visible';
                badge1.setAttribute('style','padding:5px; opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge2.setAttribute('style','padding:5px; opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
                badge3.setAttribute('style','padding:5px; opacity:1.0; -moz-opacity:1.0; filter:alpha(opacity=100)');
            }
            if(taskFlag==7 || taskFlag==8 || taskFlag==9)
            { 
                statusBtn.disabled = true; 
                statusBox.value = projectCode; 
                statusBox.style.backgroundColor = 'rgb(156, 218, 252)';
                statusBox.readOnly = true;
                mainZone.style.visibility = 'visible';
                s1Btn.style.visibility = 'hidden';
                s2Btn.style.visibility = 'hidden';
                s3Btn.style.visibility = 'hidden';
                if(taskFlag!=7)
                { 
                    badge1.setAttribute('style','padding:5px; opacity:0.5; -moz-opacity:0.5; filter:alpha(opacity=50)');
                }
                if(taskFlag!=8)
                { 
                    badge2.setAttribute('style','padding:5px; opacity:0.5; -moz-opacity:0.5; filter:alpha(opacity=50)');
                }
                if(taskFlag!=9)
                { 
                    badge3.setAttribute('style','padding:5px; opacity:0.5; -moz-opacity:0.5; filter:alpha(opacity=50)');
                }
            }
            if(taskFlag==10)
            { 
                statusBtn.disabled = true; 
                statusBox.style.backgroundColor = 'rgb(156, 218, 252)';
                statusBox.readOnly = true;
                mainZone.style.visibility = 'visible';
                s1Btn.style.visibility = 'hidden';
                s2Btn.style.visibility = 'hidden';
                s3Btn.style.visibility = 'hidden';
            }
            if(taskFlag==11)
            { 
                statusBtn.disabled = true; 
                statusBox.readOnly = true;
			    if (statusBox.style.backgroundColor.toString() == 'rgb(255, 148, 48)')
    			{ 
            		statusBox.style.backgroundColor = 'rgb(255, 110, 48)';
			    } else { 
                    statusBox.style.backgroundColor = 'rgb(255, 148, 48)';
                }                
                mainZone.style.visibility = 'visible';
                s1Btn.style.visibility = 'hidden';
                s2Btn.style.visibility = 'hidden';
                s3Btn.style.visibility = 'hidden';
            }
            if(taskFlag==12)
            { 
                statusBtn.disabled = true; 
                statusBox.read1Only = true;
			    if (statusBox.style.backgroundColor.toString() == 'rgb(249, 255, 102)')
    			{ 
            		statusBox.style.backgroundColor = 'rgb(232, 239, 33)';
			    } else { 
                    statusBox.style.backgroundColor = 'rgb(249, 255, 102)';
                }                
                mainZone.style.visibility = 'visible';
                s1Btn.style.visibility = 'hidden';
                s2Btn.style.visibility = 'hidden';
                s3Btn.style.visibility = 'hidden';
            }
            if(taskFlag==13)
            { 
                statusBtn.disabled = true; 
                statusBox.style.backgroundColor = 'rgb(132, 255, 138)';
                statusBox.readOnly = true;
                mainZone.style.visibility = 'visible';
                s1Btn.style.visibility = 'hidden';
                s2Btn.style.visibility = 'hidden';
                s3Btn.style.visibility = 'hidden';
                badge1.setAttribute('style','padding:5px; opacity:0.5; -moz-opacity:0.5; filter:alpha(opacity=50)');
                badge2.setAttribute('style','padding:5px; opacity:0.5; -moz-opacity:0.5; filter:alpha(opacity=50)');
                badge3.setAttribute('style','padding:5px; opacity:0.5; -moz-opacity:0.5; filter:alpha(opacity=50)');
            }
            
            msgBox.innerHTML = '('+taskNo.toString()+') ' + taskMessage; 
        }
         
         function submitCodeForm(oFormElement) {
            setTaskFlag(1);
            var localCode = document.getElementById('dacode').value.toString();
            var params = '';
            if (localCode != codeDefault) 
            {
                projectCode = '';
                params = 'action=check&code=' + localCode;
                taskMessage = 'here is the project code to check: '+localCode;
            }
            else
            {
                projectCode = '';
                taskMessage = 'new code to create';
                params = 'action=new&code=';
            }
            //create a task
            var xhr = new XMLHttpRequest();
            xhr.onload = function() {
                taskId = xhr.responseText;
                taskMessage = 'task id : ' + taskId;
                setTaskFlag(2);
            }
            xhr.open ('GET', 'cTaskCreate.php?'+params, true);
            xhr.send ();
            return false;
         }

         function submitSelectForm(oFormElement, side) {
            setTaskFlag(parseInt(side)+6);
            projectSide = side;
            var xhr = new XMLHttpRequest();
            xhr.onload = function() {
                taskId = xhr.responseText;
                taskMessage = 'task id : ' + taskId;
                setTaskFlag(10);
            }
            var params = 'action=select&code='+projectCode+'&side='+side;
            xhr.open ('GET', 'cTaskCreate.php?'+params, true);
            xhr.send ();
            return false;
         }
         
        function semaphor()
		{
            //document.getElementById('message').innerHTML = '('+taskNo.toString()+') '+taskMessage + ' ' + taskFlag + ' ' + projectStep  ; 
            taskNo++; 

            if ([1,2,3,4].indexOf(taskFlag) >= 0)
            {
                var xhr = new XMLHttpRequest();
                taskMessage = 'task id : ' + taskId + ' checking...';
                xhr.onload = function() {
                    var localStatus = xhr.responseText;
                    taskMessage = 'task id : ' + taskId + ' waiting...';
                    if(localStatus=='0') {setTaskFlag(4); }
                    if(localStatus=='1') {setTaskFlag(3); }
                    if(localStatus=='2') {setTaskFlag(5); }
                }
                xhr.open ('GET', 'cTaskCheck.php?id='+taskId, true);
                xhr.send ('');
            }
            if ([10,11,12].indexOf(taskFlag) >= 0)
            {
                var xhr = new XMLHttpRequest();
                taskMessage = 'task id : ' + taskId + ' checking...';
                xhr.onload = function() {
                    taskMessage = 'task id : ' + taskId + ' waiting...';
                    var localStatus = xhr.responseText;
                    if(localStatus=='0') {setTaskFlag(12); }
                    if(localStatus=='1') {setTaskFlag(11); }
                    if(localStatus=='2') {setTaskFlag(13); }
                }
                xhr.open ('GET', 'cTaskCheck.php?id='+taskId, true);
                xhr.send ('');
            }
            if (taskFlag == 5 || taskFlag == 13)
            {
                var xhr = new XMLHttpRequest();
                taskMessage = 'task id : ' + taskId + ' feedback';
                xhr.onload = function() {
                    var feedback = xhr.responseText;
                    projectCode = feedback.split('_')[0];
                    projectStep = feedback.split('_')[1];               
                    projectScore = feedback.split('_')[2];               
                    taskMessage = 'project code : ' + projectCode + ' v.'+ projectStep + ' score:'+ projectScore;
                    setTaskFlag(6);
                    cleanTask(); 
                }
                xhr.open ('GET', 'cTaskRead.php?id='+taskId, true);
                xhr.send ('');
            }            
        }

        function cleanTask()
        {
            taskMessage = 'task id : ' + taskId + ' done...';
            var xhr = new XMLHttpRequest();
            xhr.open ('GET', 'cTaskClean.php?id='+taskId, true);
            xhr.send ('');
            taskId = '';
            taskMessage = '...';
        }


    	setInterval(semaphor, 2222);
		
		</script>

    </head>
    <body>
        <div class="container">
        <header>
            <table>
                <tr>
                    <td style="vertical-align:top;" width=342>
                        <h1>Portrait-Robot</h1>
			<h3><div id="score">...</div><h3>
                    </td>
                    <td rowspan=2>
                        <img id="target" src="blank.png" style="padding:5px" />
                    </td>
                </tr>
    		<tr>
                    <td style="vertical-align:bottom">
        				<form onsubmit="return submitCodeForm(this);">
                            <label for="code">Code:</label>
        					<input type="text" id="dacode" required minlength="12" maxlength="12" size="12" value="....-....-..">
                            <button type="submit" class="btn btn-default" id="s0">create or recall</button>
                        </form>
                    </td>
                </tr>
    		</table>
    		<hr />
    	</header>
    	<section id="main" style="visibility:hidden;">
            <table>
                <tr>
                    <td>
                        <img id="badge1" src="blank.png" style="padding:5px" />
                    </td>
                    <td>
                        <img id="badge2" src="blank.png" style="padding:5px" />
                    </td>
                    <td>
                        <img id="badge3" src="blank.png"  style="padding:5px" />
                    </td>
                </tr>
		<tr>
			<td style="text-align: center;">
				<form onsubmit="return submitSelectForm(this,'1');">
    				<button type="submit" class="btn btn-default" id="s1">ô</button>
				</form>
                    	</td>
			<td style="text-align: center;">
				<form onsubmit="return submitSelectForm(this,'2');">
    				<button type="submit" class="btn btn-default" id="s2">ô</button>
	                        </form>
                        </td>
			<td style="text-align: center;">
				<form onsubmit="return submitSelectForm(this,'3');">
    				<button type="submit" class="btn btn-default" id="s3">ô</button>
				</form>
                    	</td>
    		</tr>
    	    </table>
    	</section>
        <footer>
        <hr /><div id="message">...</div>
        <div style="padding-top:16"><a target=_blank href="https://github.com/jxtrbtk/portrait-robot/"><img src="github.png"/>&nbsp;sources</a></div>
        </footer>
        </div>
    </body>
</html>

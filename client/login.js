$('#login-button').click(function (event) {
	var userName=document.getElementById("userName").value;  
    var pwd=document.getElementById("pwd").value;
    if(userName=="123" &&  pwd=="456"){ 
		event.preventDefault();
		$('form').fadeOut(500); 
		$('.wrapper').addClass('form-success');
		requestFullScreen();
		setTimeout(function(){location.href="maobi.html";},2000);
		
	}
	else{
		alert("账号或密码错误");
	}
});
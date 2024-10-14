// 获取图片和音频元素
const springImage = document.getElementById('springword');
const voice = document.getElementById('voice');

// 添加点击事件监听器
springImage.addEventListener('click', function() {
    voice.currentTime = 0; // 重置音频到开始
    voice.play(); // 播放音频
});

document.querySelector('.aldum').addEventListener('click', function() {
    document.getElementById('imageInput').click();
});

document.getElementById('imageInput').addEventListener('change', function(event) {
    var file = event.target.files[0];
    console.log('选择的文件：', file);
    var reader = new FileReader();
    reader.onload = function(e) {
        console.log('图片预览URL：', e.target.result);
    };
    reader.readAsDataURL(file);
});

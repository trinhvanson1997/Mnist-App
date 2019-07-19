var canvas = document.getElementById('myCanvas');
var ctx = canvas.getContext('2d');


canvas.width = 280;
canvas.height = 280;
fill_background_color()

var mouse = {x: 0, y: 0};

canvas.addEventListener('mousemove', function (e) {
    mouse.x = e.pageX - this.offsetLeft;
    mouse.y = e.pageY - this.offsetTop;
}, false);

ctx.lineWidth = 15;
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.strokeStyle = '#FFFFFF';

function fill_background_color() {
    // ctx.rect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

canvas.addEventListener('mousedown', function (e) {
    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);

    canvas.addEventListener('mousemove', onPaint, false);
}, false);

canvas.addEventListener('mouseup', function () {
    canvas.removeEventListener('mousemove', onPaint, false);
}, false);

var onPaint = function () {
    ctx.lineTo(mouse.x, mouse.y);
    ctx.stroke();
};


document.getElementById('clear').addEventListener('click', function () {
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    fill_background_color();
    $('#result').html("Result:");
});

$('#predict').click(function () {
    // Send base64 image over ajax and process on server
    dataURL = canvas.toDataURL();

    $.ajax({
        url: '/predict',
        type: 'POST',
        data: {
            "base_64": dataURL
        },
        success: function (res) {
            console.log(res);
            res = JSON.parse(res);
            $('#result').html("Result: " + res['number']);
            img = document.getElementById('img');
            img.src = res['encode'];
            img.src.display = 'inline';
        }
    });


});

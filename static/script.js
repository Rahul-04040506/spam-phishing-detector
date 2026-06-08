window.onload = () => {
    const bars = document.querySelectorAll('.fake-bar, .real-bar');
    bars.forEach(bar => {
        let width = bar.style.width;
        bar.style.width = "0%";
        setTimeout(() => {
            bar.style.width = width;
        }, 200);
    });
};
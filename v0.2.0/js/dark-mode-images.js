function updateLogoImages(mode) {
    document.querySelectorAll('img[data-mode]').forEach(img => {
        const lightSrc = img.getAttribute('data-light');
        const darkSrc = img.getAttribute('data-dark');
        if (mode === 'slate') {
            img.src = darkSrc || lightSrc;
        } else {
            img.src = lightSrc;
        }
    });
    console.log(`Logo images updated to ${mode} mode.`);
}

// 页面加载时执行一次
let currentMode = document.body.getAttribute('data-md-color-scheme');
updateLogoImages(currentMode);

// 使用 MutationObserver 监听 data-md-color-scheme 变化
const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
        if (mutation.attributeName === 'data-md-color-scheme') {
            const newMode = document.body.getAttribute('data-md-color-scheme');
            if (newMode !== currentMode) {
                currentMode = newMode;
                updateLogoImages(currentMode);
            }
        }
    });
});

// 开始监听 body 的属性变化
observer.observe(document.body, { attributes: true });

document.addEventListener("DOMContentLoaded", () => {
    if (window.MathJax) {
        MathJax.typesetPromise();  // 重新渲染动态插入的公式
    }
});

import * as path from 'path';
import { defineConfig } from 'rspress/config';
import katex from 'rspress-plugin-katex';

export default defineConfig({
  root: path.join(__dirname, 'docs'),
  base: '/Decent-DP/',
  title: 'Decent-DP',
  description: 'Decentralized Data Parallel',
  icon: '/icon-light.png',
  logo: {
    light: '/icon-light.png',
    dark: '/icon-dark.png',
  },
  themeConfig: {
    socialLinks: [
      { icon: 'github', mode: 'link', content: 'https://github.com/WangZesen/Decent-DP' },
    ],
    lastUpdated: true,
    enableContentAnimation: true,
  },
  mediumZoom: {
    selector: '.rspress-doc img',
  },
  multiVersion: {
    default: 'v0',
    versions: ['v0'],
  },
  plugins: [
    katex({
      macros: {
        '\\f': '#1f(#2)',
      },
    })
  ]
});

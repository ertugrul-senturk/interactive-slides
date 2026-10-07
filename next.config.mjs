/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Cross-origin isolation lets decks run ML models in the browser on several threads
  // (onnxruntime-web needs SharedArrayBuffer). "credentialless" still allows fonts, CDN
  // scripts and model files from other origins.
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "credentialless" },
        ],
      },
    ];
  },
};
export default nextConfig;

/*
    cartToPolar.shader
    This layer calculates the gradients based on the luminance stored 
    in the red channel. Then it uses the gradients to calc and store the 
    magnitude and direction (0 to 180 degrees) inside the green and blue 
    channels.
*/

Shader "SVMFaceDetection/cartToPolar"
{

    Properties
    {
        _CamTex1 ("L1 Camera", 2D) = "black" {}
        _Buffer ("cartToPolar Buffer", 2D) = "black" {}
    }

    SubShader
    {
        Tags { "Queue"="Overlay+1" "ForceNoShadowCasting"="True" "IgnoreProjector"="True" }

        Pass
        {
            Lighting Off
            SeparateSpecular Off
            Fog { Mode Off }

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma fragmentoption ARB_precision_hint_fastest
            #pragma target 5.0

            #include "UnityCG.cginc"

            #define outRes float2(410., 360.)

            //RWStructuredBuffer<float4> buffer : register(u1);

            Texture2D<float4> _CamTex1;
            Texture2D<float4> _Buffer;

            static const float3x3 fX = {
                -1, 0, 1,
                -2, 0, 2,
                -1, 0, 1
            };

            static const float3x3 fY = {
                -1, -2, -1,
                 0,  0,  0,
                 1,  2,  1
            };

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            /*
                cartToPolar is a function that computes the 
                magnitude and angle of 2D vectors.

                In:
                    px - center sample pixel
                    xrange - clamp x between [xrange.x, xrange.y]
                    yrange - clamp y between [yrange.x, yrange.y]
                Out:
                    (magnitude, direction between 0 and 180 degrees)
            */
            float2 cartToPolar(uint2 px, uint2 xrange, uint2 yrange) {

                // Calculating the gradient
                int4 off = int4(clamp(px.x - 1, xrange.x, xrange.y),  //xL
                                clamp(px.x + 1, xrange.x, xrange.y),  //xH
                                clamp(px.y - 1, yrange.x, yrange.y),  //yL
                                clamp(px.y + 1, yrange.x, yrange.y)); //yH

                float3 _0 = float3(_Buffer.Load(uint3(off.x, off.z, 0.)).r,
                                   _Buffer.Load(uint3(off.x, px.y, 0.)).r,
                                   _Buffer.Load(uint3(off.x, off.w, 0.)).r);

                float3 _1 = float3(_Buffer.Load(uint3(px.x, off.z, 0.)).r,
                                   _Buffer.Load(uint3(px.x, px.y, 0.)).r,
                                   _Buffer.Load(uint3(px.x, off.w, 0.)).r);

                float3 _2 = float3(_Buffer.Load(uint3(off.y, off.z, 0.)).r,
                                   _Buffer.Load(uint3(off.y, px.y, 0.)).r,
                                   _Buffer.Load(uint3(off.y, off.w, 0.)).r);

                float4 g;
                g.x = _0.x * fX[0][0] + _0.y * fX[0][1] + _0.z * fX[0][2] +
                _1.x * fX[1][0] + _1.y * fX[1][1] + _1.z * fX[1][2] +
                _2.x * fX[2][0] + _2.y * fX[2][1] + _2.z * fX[2][2];

                g.y = _0.x * fY[0][0] + _0.y * fY[0][1] + _0.z * fY[0][2] +
                _1.x * fY[1][0] + _1.y * fY[1][1] + _1.z * fY[1][2] +
                _2.x * fY[2][0] + _2.y * fY[2][1] + _2.z * fY[2][2];

                // Magnitude
                g.z = sqrt(g.y*g.y+g.x*g.x);
                // Direction (0 to 180 degrees)
                g.w = atan2(g.y, g.x) * 180.0 / UNITY_PI;
                g.w = g.w < 0.0 ? g.w + 360.0 : g.w;
                g.w = g.w > 180.0 ? g.w - 180.0 : g.w;

                //if (g.z < 0)
                //    buffer[0] = float4(g.z, g.w, 0, 0);

                return g.zw;
            }

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float4 frag (v2f ps) : SV_Target
            {
                uint2 px = round(ps.uv * outRes);
                
                float3 col = float3(0.,0.,0.);
                float3 camTex = float3(0.,0.,0.);

                // I resize the camera input instead of the sliding window
                // Makes life easier

                if (px.y < 140) {
                    // Resize to 250x140
                    if (px.x < 250) {
                        uint2 mpx = px * 1.56;
                        camTex = _CamTex1.Load(uint3(mpx, 0));
                        col.gb = cartToPolar(px, uint2(0, 249), uint2(0, 139));
                    }
                    else {
                        if (px.y < 50) discard;
                        // Resize to 160x90
                        //return float4(0,0,0,0);
                        uint2 mpx = round((px - uint2(250, 50)) * 2.4375);
                        camTex = _CamTex1.Load(uint3(mpx, 0));
                        col.gb = cartToPolar(px, uint2(250, 409), uint2(50, 139));
                    }
                }
                else {
                    // Default size 390x220
                    if (px.x < 390) {
                        uint2 mpx = uint2(px.x, px.y - 140);
                        camTex = _CamTex1.Load(uint3(mpx, 0));
                        col.gb = cartToPolar(px, uint2(0, 389), uint2(140, 359));
                    }
                }

                // Relative luminance
                col.r = 0.2126*camTex.r + 0.7152*camTex.g + 0.0722*camTex.b;
                return float4(col, 1.0);
            }

            ENDCG
        }
    }
    FallBack "Diffuse"
}

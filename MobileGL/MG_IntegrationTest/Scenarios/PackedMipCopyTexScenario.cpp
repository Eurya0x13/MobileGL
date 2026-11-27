// MobileGL - MobileGL/MG_IntegrationTest/Scenarios/PackedMipCopyTexScenario.cpp
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v3.0:
//   https://www.gnu.org/licenses/gpl-3.0.txt
//   https://www.gnu.org/licenses/lgpl-3.0.txt
// SPDX-License-Identifier: LGPL-3.0-only
// End of Source File Header

#include <MG_IntegrationTest/Harness/ScenarioFixture.h>
#include <MG_Backend/DirectGLES/Managers.h>

#ifdef GLAPI
#undef GLAPI
#endif
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glcorearb.h>
#undef GL_GLEXT_PROTOTYPES

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace MGITest {
    namespace {
        namespace GLES = MobileGL::MG_Backend::DirectGLES;
        constexpr int mipCount = 5;
        constexpr int atlasWidth = 1024;
        constexpr int atlasHeight = 512;
        using Pixels = std::vector<std::uint8_t>;

        struct Patch {
            int x, y, width, height;
        };

        // A measured atlas update. Direct uploads split the first two levels
        // and transfer 37568 bytes. The packed page transfers 12288 bytes.
        constexpr std::array<Patch, 3> atlasPatches{{{112, 96, 32, 16}, {0, 112, 32, 16}, {432, 96, 16, 64}}};

        enum class UploadPath {
            Packed,
            RejectedCopy,
            DirectSplit
        };

        constexpr const char* vertexShader = R"(#version 430 core
void main() {
    vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
}
)";
        constexpr const char* fragmentShader = R"(#version 430 core
uniform sampler2D atlas;
out vec4 color;
void main() {
    color = texelFetch(atlas, ivec2(0, 0), 0);
}
)";

        std::array<GLint, 4> DriverViewport() {
            std::array<GLint, 4> box{};
            GLES::g_GLESFuncs.glGetIntegerv(GL_VIEWPORT, box.data());
            return box;
        }

        GLint DriverInteger(GLenum name) {
            GLint value = 0;
            GLES::g_GLESFuncs.glGetIntegerv(name, &value);
            return value;
        }

        struct NativeHooks {
            explicit NativeHooks(bool reject) : reject(reject) {
                active = this;
                GLES::g_GLESFuncs.glCopyTexSubImage2D = CopyTex;
                GLES::g_GLESFuncs.glTexSubImage2D = Upload;
            }

            ~NativeHooks() {
                GLES::g_GLESFuncs.glCopyTexSubImage2D = copyTex;
                GLES::g_GLESFuncs.glTexSubImage2D = upload;
                active = nullptr;
            }

            static void GL_APIENTRY CopyTex(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y,
                                            GLsizei width, GLsizei height) {
                EXPECT_EQ(target, GLenum{GL_TEXTURE_2D});
                EXPECT_EQ(DriverInteger(GL_DRAW_FRAMEBUFFER_BINDING), active->drawFramebuffer);
                EXPECT_EQ(DriverInteger(GL_READ_BUFFER), GLint{GL_COLOR_ATTACHMENT0});
                EXPECT_NE(DriverInteger(GL_READ_FRAMEBUFFER_BINDING), active->readFramebuffer);
                EXPECT_NE(DriverInteger(GL_TEXTURE_BINDING_2D), active->stagingTexture);
                GLint source = 0;
                GLES::g_GLESFuncs.glGetFramebufferAttachmentParameteriv(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                                                        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &source);
                EXPECT_EQ(source, active->stagingTexture);
                active->levels.push_back(level);
                if (active->reject && active->levels.size() == 2) width = -1;
                active->copyTex(target, level, xoffset, yoffset, x, y, width, height);
            }

            static void GL_APIENTRY Upload(GLenum target, GLint level, GLint x, GLint y, GLsizei width, GLsizei height,
                                           GLenum format, GLenum type, const void* data) {
                if (++active->uploads == 1) active->stagingTexture = DriverInteger(GL_TEXTURE_BINDING_2D);
                active->uploadBytes += 4 * width * height;
                active->upload(target, level, x, y, width, height, format, type, data);
            }

            inline static NativeHooks* active = nullptr;
            const decltype(GLES::g_GLESFuncs.glCopyTexSubImage2D) copyTex = GLES::g_GLESFuncs.glCopyTexSubImage2D;
            const decltype(GLES::g_GLESFuncs.glTexSubImage2D) upload = GLES::g_GLESFuncs.glTexSubImage2D;
            const bool reject;
            GLint stagingTexture = 0;
            GLint readFramebuffer = 0;
            GLint drawFramebuffer = 0;
            int uploads = 0;
            int uploadBytes = 0;
            std::vector<int> levels;
        };

        class PackedMipCopyTexScenario : public ScenarioTest {
        protected:
            void Verify(UploadPath path) {
                if (Gl().BackendName() != "DirectGLES" || SplitLane::IsSplitLane()) {
                    GTEST_SKIP() << "requires in-process native GLES call inspection";
                }
                if (!GLES::BufferImpl::UnpackRingAvailable()) {
                    GTEST_SKIP() << "the native driver does not provide the persistent unpack ring";
                }
                std::array<Pixels, mipCount> expected;
                GLuint atlas = 0;
                glGenTextures(1, &atlas);
                glBindTexture(GL_TEXTURE_2D, atlas);
                glTexStorage2D(GL_TEXTURE_2D, mipCount, GL_RGBA8, atlasWidth, atlasHeight);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, mipCount - 1);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                for (int level = 0; level < mipCount; ++level) {
                    const int width = atlasWidth >> level;
                    const int height = atlasHeight >> level;
                    expected[level].resize(width * height * 4);
                    for (int i = 0; i < width * height; ++i) {
                        expected[level][i * 4] = i % 7;
                        expected[level][i * 4 + 1] = level + 4;
                        expected[level][i * 4 + 2] = i % 13;
                        expected[level][i * 4 + 3] = 157;
                    }
                    glTexSubImage2D(GL_TEXTURE_2D, level, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                                    expected[level].data());
                }

                const auto first = MakeColorFbo(16, 16);
                const auto second = MakeColorFbo(16, 16);
                ASSERT_NE(first.fbo, 0u);
                ASSERT_NE(second.fbo, 0u);
                std::string error;
                const GLuint program = CompileProgram(vertexShader, fragmentShader, &error);
                ASSERT_NE(program, 0u) << error;
                GLuint vao = 0;
                glGenVertexArrays(1, &vao);
                glBindVertexArray(vao);
                glUseProgram(program);
                glUniform1i(glGetUniformLocation(program, "atlas"), 0);
                GLuint sampler = 0;
                glGenSamplers(1, &sampler);
                glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
                glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glSamplerParameteri(sampler, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
                glBindSampler(0, sampler);
                glBindTexture(GL_TEXTURE_2D, atlas);
                BindFbo(first);
                glDrawArrays(GL_TRIANGLES, 0, 3);
                glFinish();
                ASSERT_EQ(FirstGLError(), GLenum{GL_NO_ERROR});

                // Initialize the unpack ring with an ordinary upload to existing storage.
                BindFbo(second);
                glClear(GL_COLOR_BUFFER_BIT);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, atlasWidth, 128, GL_RGBA, GL_UNSIGNED_BYTE, expected[0].data());
                glDrawArrays(GL_TRIANGLES, 0, 3);
                ASSERT_EQ(FirstGLError(), GLenum{GL_NO_ERROR});

                // A different draw FBO makes these updates eligible for packed staging.
                glBindFramebuffer(GL_DRAW_FRAMEBUFFER, first.fbo);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, second.fbo);
                glEnable(GL_SCISSOR_TEST);
                glScissor(0, 0, 1, 1);
                glViewport(1, 2, 13, 14);
                glEnable(GL_BLEND);
                glBlendFunc(GL_ZERO, GL_ONE);
                glColorMask(GL_TRUE, GL_FALSE, GL_TRUE, GL_FALSE);
                glEnable(GL_CULL_FACE);
                glCullFace(GL_FRONT_AND_BACK);
                glEnable(GL_RASTERIZER_DISCARD);
                glEnable(GL_DITHER);
                glClear(GL_COLOR_BUFFER_BIT);
                glReadBuffer(GL_NONE);
                // The application read buffer is applied lazily. Set the native
                // selection on the read FBO before testing its restoration.
                GLES::g_GLESFuncs.glReadBuffer(GL_NONE);
                const GLint initialDraw = DriverInteger(GL_DRAW_FRAMEBUFFER_BINDING);
                const GLint initialRead = DriverInteger(GL_READ_FRAMEBUFFER_BINDING);
                ASSERT_NE(initialDraw, initialRead);
                ASSERT_NE(initialRead, 0);
                const auto initialProgram = DriverInteger(GL_CURRENT_PROGRAM);
                const auto initialVao = DriverInteger(GL_VERTEX_ARRAY_BINDING);
                const auto initialSampler = DriverInteger(GL_SAMPLER_BINDING);
                const auto initialViewport = DriverViewport();
                ASSERT_EQ(DriverInteger(GL_READ_BUFFER), GLint{GL_NONE});

                const bool cheapSplit = path == UploadPath::DirectSplit;
                for (int level = 0; level < mipCount; ++level) {
                    const int levelWidth = atlasWidth >> level;
                    const int levelHeight = atlasHeight >> level;
                    auto patches = atlasPatches;
                    for (auto& patch : patches) {
                        patch = {patch.x >> level, patch.y >> level, patch.width >> level, patch.height >> level};
                    }
                    if (cheapSplit) {
                        // Two widely separated patches in the largest levels fit
                        // two cheap direct boxes. Smaller levels stay compact.
                        patches[0] = {1, 0, 1, 12};
                        patches[1] = level < 2 ? Patch{levelWidth - 13, levelHeight - 9, 13, 9} : Patch{20, 0, 13, 9};
                    }
                    const int patchCount = cheapSplit ? 2 : 3;
                    for (int patch = 0; patch < patchCount; ++patch) {
                        const auto [x0, y0, width, height] = patches[patch];
                        Pixels patchData(width * height * 4);
                        for (int y = 0; y < height; ++y) {
                            for (int x = 0; x < width; ++x) {
                                const auto at = (y * width + x) * 4;
                                const auto dst = ((y0 + y) * levelWidth + x0 + x) * 4;
                                patchData[at] = 101 + x + patch * 15;
                                patchData[at + 1] = 131 + y + level * 13;
                                patchData[at + 2] = 214 + patch;
                                patchData[at + 3] = 72 + patch * 7 + level * 11;
                                std::copy_n(patchData.data() + at, 4, expected[level].data() + dst);
                            }
                        }
                        glTexSubImage2D(GL_TEXTURE_2D, level, x0, y0, width, height, GL_RGBA, GL_UNSIGNED_BYTE,
                                        patchData.data());
                    }
                }

                {
                    NativeHooks hooks{path == UploadPath::RejectedCopy};
                    hooks.drawFramebuffer = initialDraw;
                    hooks.readFramebuffer = initialRead;
                    glDrawArrays(GL_TRIANGLES, 0, 3);
                    if (cheapSplit) {
                        EXPECT_TRUE(hooks.levels.empty());
                        EXPECT_EQ(hooks.uploads, 7);
                        // Two exact rects in each of two levels, one compact
                        // union in each of the three smaller levels.
                        EXPECT_EQ(hooks.uploadBytes, 4 * (2 * (12 + 13 * 9) + 3 * 32 * 12));
                    } else {
                        EXPECT_EQ(hooks.levels.size(), 15u);
                        for (int level = 0; level < mipCount; ++level) {
                            EXPECT_EQ(std::count(hooks.levels.begin(), hooks.levels.end(), level), 3);
                        }
                        if (path == UploadPath::RejectedCopy) {
                            EXPECT_EQ(hooks.uploads, 8);
                            EXPECT_EQ(hooks.uploadBytes, 12288 + 37568);
                        } else {
                            EXPECT_EQ(hooks.uploads, 1);
                            EXPECT_EQ(hooks.uploadBytes, 12288);
                        }
                    }
                }

                EXPECT_EQ(DriverInteger(GL_READ_FRAMEBUFFER_BINDING), initialRead);
                EXPECT_EQ(DriverInteger(GL_DRAW_FRAMEBUFFER_BINDING), initialDraw);
                EXPECT_EQ(DriverInteger(GL_READ_BUFFER), GLint{GL_NONE});
                EXPECT_EQ(DriverInteger(GL_CURRENT_PROGRAM), initialProgram);
                EXPECT_EQ(DriverInteger(GL_VERTEX_ARRAY_BINDING), initialVao);
                EXPECT_EQ(DriverInteger(GL_SAMPLER_BINDING), initialSampler);
                EXPECT_EQ(DriverViewport(), initialViewport);
                EXPECT_EQ(GLES::g_GLESFuncs.glIsEnabled(GL_SCISSOR_TEST), GL_TRUE);
                EXPECT_EQ(GLES::g_GLESFuncs.glIsEnabled(GL_BLEND), GL_TRUE);
                EXPECT_EQ(GLES::g_GLESFuncs.glIsEnabled(GL_CULL_FACE), GL_TRUE);
                EXPECT_EQ(GLES::g_GLESFuncs.glIsEnabled(GL_RASTERIZER_DISCARD), GL_TRUE);
                std::array<GLboolean, 4> colorMask{};
                GLES::g_GLESFuncs.glGetBooleanv(GL_COLOR_WRITEMASK, colorMask.data());
                const std::array<GLboolean, 4> expectedMask{GL_TRUE, GL_FALSE, GL_TRUE, GL_FALSE};
                EXPECT_EQ(colorMask, expectedMask);
                EXPECT_EQ(FirstGLError(), GLenum{GL_NO_ERROR});

                // glReadPixels from each mip FBO reads native texture storage and also checks unchanged texels.
                GLuint readback = 0;
                glGenFramebuffers(1, &readback);
                glBindFramebuffer(GL_READ_FRAMEBUFFER, readback);
                glReadBuffer(GL_COLOR_ATTACHMENT0);
                glPixelStorei(GL_PACK_ALIGNMENT, 1);
                for (int level = 0; level < mipCount; ++level) {
                    const int width = atlasWidth >> level;
                    const int height = atlasHeight >> level;
                    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, atlas, level);
                    ASSERT_EQ(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER), GLenum{GL_FRAMEBUFFER_COMPLETE});
                    Pixels actual(width * height * 4);
                    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, actual.data());
                    const auto firstDifference = std::mismatch(actual.begin(), actual.end(), expected[level].begin());
                    EXPECT_TRUE(firstDifference.first == actual.end())
                        << "mip " << level << ", first byte " << std::distance(actual.begin(), firstDifference.first);
                }
                EXPECT_EQ(FirstGLError(), GLenum{GL_NO_ERROR});

                glUseProgram(0);
                glDeleteProgram(program);
                glDeleteVertexArrays(1, &vao);
                glBindSampler(0, 0);
                glDeleteSamplers(1, &sampler);
                glDisable(GL_SCISSOR_TEST);
                glDisable(GL_BLEND);
                glBlendFunc(GL_ONE, GL_ZERO);
                glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                glDisable(GL_CULL_FACE);
                glCullFace(GL_BACK);
                glDisable(GL_RASTERIZER_DISCARD);
                glPixelStorei(GL_PACK_ALIGNMENT, 4);
                glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
                BindDefaultFramebuffer();
                glDeleteFramebuffers(1, &readback);
                auto firstFbo = first;
                auto secondFbo = second;
                DestroyColorFbo(firstFbo);
                DestroyColorFbo(secondFbo);
                glDeleteTextures(1, &atlas);
                glFinish();
                Gl().EndFrame();
                glFinish();
                Gl().EndFrame();
            }
        };

        TEST_F(PackedMipCopyTexScenario, CopiesFiveMipsAndRestoresDriverState) {
            if (Ready()) Verify(UploadPath::Packed);
        }

        TEST_F(PackedMipCopyTexScenario, PartialCopyFailureReplaysPixelsAndRestoresDriverState) {
            if (Ready()) Verify(UploadPath::RejectedCopy);
        }

        TEST_F(PackedMipCopyTexScenario, PrefersTwoCheapDirectBoxesOverPackedCopy) {
            if (Ready()) Verify(UploadPath::DirectSplit);
        }
    } // namespace
} // namespace MGITest

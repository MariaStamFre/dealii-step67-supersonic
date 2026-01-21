# -------- build stage --------
FROM dealii/dealii:v9.3.0-focal AS builder

WORKDIR /build
COPY *.cc *.hpp CMakeLists.txt .
RUN cmake . && make -j$(nproc)

# -------- run stage --------
FROM dealii/dealii:v9.3.0-focal

WORKDIR /home/dealii_user/step-67-supersonic
COPY *.cc *.hpp CMakeLists.txt .
COPY --from=builder /build/step-67 .

RUN mkdir results

CMD ["mpirun", "-np", "4", "./step-67"]

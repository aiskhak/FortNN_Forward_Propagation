! Fortran program that reads weights and biases 
! from a file and performs a forward propagation

program forward_pass

   ! module with neural network structure
   use fortfp

   implicit none
   
   ! types
   type (nn) :: mynn                                ! neural network
   type (af), allocatable, dimension(:) :: afuncs   ! activation functions

   ! variables, arrays, and matrices
   real(8), allocatable, dimension (:,:) :: inputs  ! array of inputs: (n_neurons, batch_size)
   real(8), allocatable, dimension (:) :: outputs   ! array of outputs: (n_neurons)
   integer nl                                       ! number of layers in neural network
   integer n_in, n_hi, n_out			    		       ! number of neurons
   integer n_batch			            			    ! input batch size
   integer i, j                                     ! loop parameters
   
   ! input parameters
   nl = 6
   n_in = 9
   n_out = 6
   n_hi = 100
   n_batch = 1

   ! allocate memory
   allocate(afuncs(nl), inputs(n_in, n_batch), outputs(n_out))

   ! generate some inputs for forward propagation
   do j = 1,n_in
      do i = 1,n_batch
         inputs(j,i) = 0.d0
      enddo
   enddo

   ! define activation functions in each layer
   !    options: htan, lrelu, smax, sigmoid, no_activation
   afuncs(1)%f => null()   ! no activation
   afuncs(2)%f => lrelu
   afuncs(3)%f => lrelu
   afuncs(4)%f => lrelu
   afuncs(5)%f => lrelu
   afuncs(nl)%f => no_activation

   ! define number of neurons in each layer and initialize neural network
   call mynn%init([n_in, n_hi, n_hi, n_hi, n_hi, n_out], afuncs)

   ! perform forward pass for each input
   do i = 1,n_batch
      
      call mynn%query(inputs(:,i))
      outputs = mynn%layers(nl-1)%y   ! layers' numbers begin from 0
      print *, "inputs: ", inputs(:,i)
      print *, "outputs: ", outputs

   enddo
   
end program forward_pass

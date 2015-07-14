$(document).ready(function(){
    $('section#screenshots a').on('click', function(){
        $('div#modal img').attr('src', $(this).attr('data-image-url'));
    });
  
    var nav = $('.navbar-fixed-top');
    var distance = $('.navbar-fixed-top').offset();
  
    if (distance.top >= 300) {
      nav.addClass('effect');
    }
    
    
    $(window).scroll(function() {
      var scroll = $(window).scrollTop();
      
      if(scroll >= 300) {
        nav.addClass('effect'); 
      } else {        
        nav.removeClass('effect');        
      }
      
    });
  
     $('#about .blue-circle').waypoint(function(){
      $('#about .blue-circle').addClass('animated fadeInUp');   
    }, {
      offset: '50%'
      
    });
  
    $('#features .blue-circle').waypoint(function(){
      $(this.element).addClass('animated fadeInUp');   
    }, {
      offset: '50%'
      
    });
    
    $('.features-image img').waypoint(function(){
      $('.features-image img').addClass('animated rubberBand');  
    }, {
      offset: '50%'
      
    });
    
    $('#screenshots .col-xs-4').waypoint(function(){
      $(this.element).addClass('animated zoomIn');
      $(this.element).css({'opacity':1});
    }, {
      offset: '50%'
      
    });
  
    $('#download div.phone img').waypoint(function(){
      $(this.element).addClass('animated fadeInRight');
      
    }, {
      offset: '50%'
      
    });
    $('#download .platforms > div').waypoint(function(){
      $(this.element).addClass('animated fadeInUp');
      
    }, {
      offset: '50%'
      
    });
  
    $('#form').bootstrapValidator({
      message: 'The value is not valid',
      feedbackIcons: {
        valid: 'glyphicon glyphicon-ok',
        validating:'glyphicon glyphicon-refresh',
        invalid:'glyphicon glyphicon-remove',
      },
      fields:{
        name: {
          validators: {
            notEmpty: {
              message:'This field is required and cannot be empty'
            }
          }
        },
          email:{
                validators: {
                  notEmpty: {
                    message: 'This field is required and cannot be empty'
            },
            emailAddress: {
              message: 'The input is not a valid address'
            }
          }
        },
        message:{
          validators: {
            notEmpty:{
              message: 'The message cannot be empty'
            }
          }
        }
       }
     }).on('success.form.bv' , function(e){
      e.preventDefault();
      
      var $form =$(e.target);
      var bv=$form.data('bootstrapValidator');
      $.post($form.attr('action'),$form.serialize(), function(result) {
        
        console.log(result);
      }, 'json' );
        
    });
              
  
});
  
smoothScroll.init({
  speed: 700,
  easing: 'easeInOutQuad',
  updateURL:false,
  offset:0

});